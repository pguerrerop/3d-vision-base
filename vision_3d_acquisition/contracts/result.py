from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from vision_3d_acquisition.contracts.modalities import CaptureModality


Decision = Literal["accept", "reject", "review"]
ProcessingStatus = Literal["ok", "failed"]
ProcessingMode = Literal["mock", "real"]
ProcessingEngine = Literal["legacy", "native", "mock"]
PlaneMode = Literal["calibrated", "auto", "disabled"]
CalibrationResolutionSource = Literal["explicit", "runtime_default", "none"]
AlgorithmStage = Literal["mock", "segmentation", "calibrated_segmentation", "classification", "production"]
ObjectClass = Literal["ball", "non_ball", "unknown"]
ArtifactKind = Literal["image", "point_cloud", "table", "json", "metric", "overlay", "video", "text", "file"]
OverlayType = Literal["bbox", "mask", "ellipse", "footprint_roundness", "centroid", "polyline", "text", "segmentation"]
OverlayCoordinateSpace = Literal["image_pixel", "normalized_image", "projection_pixel", "world_mm", "plot_pixel", "point_cloud_projection"]
ProjectionType = Literal["xy_topdown", "xz_side", "yz_side", "object_crop", "heightmap", "depthmap"]
StageExecutionStatus = Literal["success", "warning", "failed", "skipped"]
CandidateSourceModality = Literal["rgb", "heightmap", "point_cloud", "reflectance", "derived_25d"]


class ResultSummary(BaseModel):
    object_count: int = Field(ge=0)
    ball_count: int = Field(ge=0)
    non_ball_count: int = Field(ge=0)
    decision: Decision
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None
    superclass: str | None = None
    feature_runtime_summary: dict[str, Any] = Field(default_factory=dict)
    feature_readiness: dict[str, Any] = Field(default_factory=dict)


class DetectedObject(BaseModel):
    object_id: int
    class_name: ObjectClass
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None
    display_label: str | None = None
    superclass: str | None = None
    class_group: str | None = None
    point_count: int | None = Field(default=None, ge=0)
    center_mm: tuple[float, float, float] | None = None
    dimensions_mm: tuple[float, float, float] | None = None
    diameter_estimate_mm: float | None = None
    bbox_min_mm: tuple[float, float, float] | None = None
    bbox_max_mm: tuple[float, float, float] | None = None
    diameter_mm: float | None = None
    sphericity_score: float | None = None
    fit_rmse_mm: float | None = None
    height_above_belt_mm: dict[str, float] | None = None
    fraction_points_inside_belt: float | None = None
    filter_status: Literal["kept", "rejected"] | None = None
    filter_reason: str | None = None
    feature_volume_proxy_mm3: float | None = None
    feature_flatness: float | None = None
    feature_eccentricity: float | None = None
    feature_sphericity_3d: float | None = None
    feature_edge_roughness: float | None = None
    feature_local_curvature_proxy: float | None = None
    feature_height_asymmetry: float | None = None
    feature_footprint_roundness: float | None = None
    footprint_geometry: dict[str, Any] | None = None
    surface_geometry: dict[str, Any] | None = None
    sphere_consistency: dict[str, Any] | None = None
    damage_metrics: dict[str, Any] | None = None
    feature_group_summaries: list[dict[str, Any]] = Field(default_factory=list)
    feature_warnings: list[dict[str, Any]] = Field(default_factory=list)
    feature_readiness: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)


class ProcessingArtifact(BaseModel):
    artifact_id: str
    stage_id: str
    object_id: int | None = None
    kind: ArtifactKind
    title: str
    description: str | None = None
    path: str | None = None
    mime_type: str | None = None
    preview_available: bool = False
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated: Literal["explicit", "derived"] = "explicit"
    overlay_type: OverlayType | None = None
    coordinate_space: OverlayCoordinateSpace | None = None
    target_artifact_id: str | None = None
    geometry: dict[str, Any] = Field(default_factory=dict)
    style: dict[str, Any] = Field(default_factory=dict)
    source_artifact_ids: list[str] = Field(default_factory=list)
    approximate: bool = False
    overlay_warnings: list[str] = Field(default_factory=list)
    projection_type: ProjectionType | None = None
    projection_coordinate_system: dict[str, Any] = Field(default_factory=dict)
    projection_metadata: dict[str, Any] = Field(default_factory=dict)
    projection_transform_id: str | None = None


class StageExecutionReport(BaseModel):
    stage_id: str
    status: StageExecutionStatus
    started_at: float | None = Field(default=None, ge=0)
    finished_at: float | None = Field(default=None, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    object_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)


class PipelineExecutionTrace(BaseModel):
    pipeline_id: str | None = None
    stages: list[StageExecutionReport] = Field(default_factory=list)


class ProcessingUnitMetadata(BaseModel):
    stage_id: str
    display_name: str
    version: str = "1.0"
    description: str = ""
    required_modalities: list[CaptureModality] = Field(default_factory=list)
    optional_modalities: list[CaptureModality] = Field(default_factory=list)
    produced_artifact_kinds: list[ArtifactKind] = Field(default_factory=list)
    object_outputs: bool = False
    supports_real_time: bool = False
    dependencies: list[str] = Field(default_factory=list)
    optional_stage: bool = False
    condition: str | None = None


class ProjectionCoordinateSystem(BaseModel):
    origin: tuple[float, float, float] | None = None
    x_axis: tuple[float, float, float] | None = None
    y_axis: tuple[float, float, float] | None = None
    z_axis: tuple[float, float, float] | None = None
    pixel_per_mm: float | None = None
    world_bounds_mm: dict[str, float] = Field(default_factory=dict)
    image_width: int | None = None
    image_height: int | None = None
    affine_transform: list[list[float]] = Field(default_factory=list)


class ProjectionArtifactMetadata(BaseModel):
    projection_type: ProjectionType
    coordinate_system: ProjectionCoordinateSystem
    source_point_cloud_id: str | None = None
    source_object_id: int | None = None
    background_style: str | None = None
    colormap: str | None = None
    depth_range: tuple[float, float] | None = None
    transform_id: str | None = None


class ResultFiles(BaseModel):
    point_cloud: str | None = None
    point_cloud_npz: str | None = None
    heightmap: str | None = None
    heightmap_npz: str | None = None
    normalized_heightmap: str | None = None
    classification_explanation: str | None = None
    metric_explanation: str | None = None
    heightmap_metadata: str | None = None
    reflectance: str | None = None
    input_preview: str | None = None
    overlay: str | None = None
    debug_height: str | None = None
    debug_segmentation: str | None = None
    debug_plane_segmentation: str | None = None
    debug_foreground: str | None = None
    debug_clusters: str | None = None
    debug_calibrated_planes: str | None = None
    debug_belt_polygon_topview: str | None = None
    debug_filtered_foreground: str | None = None
    debug_rejected_points: str | None = None
    debug_clusters_filtered: str | None = None
    measurement_diagnostics: str | None = None
    feature_vector: str | None = None
    feature_provenance: str | None = None
    quality_flags: str | None = None
    feature_runtime_summary: str | None = None
    feature_studio_summary: str | None = None


class PointCloudStats(BaseModel):
    point_count: int = Field(ge=0)
    has_colors: bool
    has_normals: bool
    min_bound: tuple[float, float, float]
    max_bound: tuple[float, float, float]
    extent: tuple[float, float, float]
    file_size_bytes: int | None = Field(default=None, ge=0)


class ProcessingTiming(BaseModel):
    load: int = Field(ge=0)
    segmentation: int = Field(ge=0)
    classification: int = Field(ge=0)
    total: int = Field(ge=0)


ProfilingCategory = Literal[
    "input",
    "io",
    "calibration",
    "preprocessing",
    "calibration_filtering",
    "segmentation",
    "clustering",
    "geometry",
    "measurement",
    "classification",
    "overlay",
    "debug_artifacts",
    "output",
]


class ProcessingProfileStage(BaseModel):
    name: str
    category: ProfilingCategory
    started_at: float = Field(ge=0)
    ended_at: float | None = Field(default=None, ge=0)
    duration_ms: float = Field(ge=0)
    input_points: int | None = Field(default=None, ge=0)
    output_points: int | None = Field(default=None, ge=0)
    input_objects: int | None = Field(default=None, ge=0)
    output_objects: int | None = Field(default=None, ge=0)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingProfiling(BaseModel):
    total_ms: float = Field(ge=0)
    production_ms: float = Field(ge=0)
    debug_artifacts_ms: float = Field(ge=0)
    io_ms: float = Field(ge=0)
    stages: list[ProcessingProfileStage] = Field(default_factory=list)


class ProcessingPipelineInfo(BaseModel):
    name: str
    id: str | None = None
    display_name: str | None = None
    pipeline_family: str | None = None
    required_modalities: list[CaptureModality] = Field(default_factory=list)
    optional_modalities: list[CaptureModality] = Field(default_factory=list)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    processing_units: list[dict[str, Any]] = Field(default_factory=list)
    composition: dict[str, Any] = Field(default_factory=dict)


class StageOutput(BaseModel):
    stage: str
    display_name: str | None = None
    implemented: bool = True
    artifacts: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)


class ResultCalibrationInfo(BaseModel):
    calibration_type: str = "plane_3d"
    source_modalities: list[CaptureModality] = Field(default_factory=lambda: ["point_cloud"])
    resolution_source: CalibrationResolutionSource | None = None
    file: str | None = None
    active: bool = False


class RuntimeAcquisitionMetrics(BaseModel):
    acquisition_connected: bool | None = None
    acquisition_source: str | None = None
    latest_frame_timestamp: str | None = None
    dropped_frames: int | None = Field(default=None, ge=0)
    queue_size: int | None = Field(default=None, ge=0)
    processing_lag_ms: float | None = Field(default=None, ge=0.0)
    queue_wait_ms: float | None = Field(default=None, ge=0.0)
    last_successful_processing_at: str | None = None


class ThroughputMetrics(BaseModel):
    acquisition_fps: float | None = Field(default=None, ge=0.0)
    processing_fps: float | None = Field(default=None, ge=0.0)
    average_processing_latency_ms: float | None = Field(default=None, ge=0.0)
    max_processing_latency_ms: float | None = Field(default=None, ge=0.0)
    average_queue_wait_ms: float | None = Field(default=None, ge=0.0)
    render_debug_overhead_ms: float | None = Field(default=None, ge=0.0)
    export_write_overhead_ms: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class ObjectCandidate(BaseModel):
    id: str
    local_object_index: int = Field(ge=1)
    source_modality: CandidateSourceModality
    source_take_id: str
    source_pipeline_run_id: str | None = None
    acquisition_group_id: str | None = None
    bbox_px: list[float] | None = None
    contour_px: list[list[float]] | None = None
    mask_artifact_id: str | None = None
    overlay_artifact_id: str | None = None
    centroid_px: list[float] | None = None
    centroid_world: list[float] | None = None
    geometry: dict[str, Any] = Field(default_factory=dict)
    appearance: dict[str, Any] = Field(default_factory=dict)
    measurements: dict[str, Any] = Field(default_factory=dict)
    classification_hints: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    take_id: str
    session_id: str | None = None
    frameset_id: str | None = None
    processed_at: datetime
    processing_mode: ProcessingMode
    processing_engine: ProcessingEngine | None = None
    plane_mode: PlaneMode | None = None
    calibration_status: str | None = None
    calibration_resolution_source: CalibrationResolutionSource | None = None
    calibration_active: bool | None = None
    input_modalities: list[CaptureModality] = Field(default_factory=lambda: ["point_cloud"])
    output_modalities: list[str] = Field(default_factory=list)
    processing_pipeline: ProcessingPipelineInfo | None = None
    stage_outputs: list[StageOutput] = Field(default_factory=list)
    pipeline_execution: PipelineExecutionTrace | None = None
    artifacts: list[ProcessingArtifact] = Field(default_factory=list)
    calibration: ResultCalibrationInfo | None = None
    algorithm_stage: AlgorithmStage
    status: ProcessingStatus
    summary: ResultSummary
    plane_model: tuple[float, float, float, float] | None = None
    input_stats: PointCloudStats | None = None
    objects: list[DetectedObject] = Field(default_factory=list)
    files: ResultFiles = Field(default_factory=ResultFiles)
    timing_ms: ProcessingTiming
    profiling: ProcessingProfiling | None = None
    error: str | None = None
    calibration_id: str | None = None
    calibration_file: str | None = None
    calibration_snapshot: dict[str, Any] | None = None
    plane_filtering: dict[str, Any] | None = None
    rejected_objects: list[DetectedObject] = Field(default_factory=list)
    poc_summary: dict[str, Any] | None = None
    calibration_diagnostics: dict[str, Any] | None = None
    runtime_acquisition: RuntimeAcquisitionMetrics | None = None
    throughput: ThroughputMetrics | None = None
    synchronization: dict[str, Any] | None = None
    acquisition_timestamps: dict[str, str] | None = None
    recipe_version_id: str | None = None
    config_snapshot_hash: str | None = None
    stage_params: dict[str, Any] = Field(default_factory=dict)
    recipe_snapshot: dict[str, Any] | None = None
    processing_unit_trace: dict[str, Any] | None = None
    source_id: str | None = None
    acquisition_group_id: str | None = None
    calibration_profile_id: str | None = None
    object_candidates: list[ObjectCandidate] = Field(default_factory=list)
    classification: dict[str, Any] | None = None

    @field_validator("take_id")
    @classmethod
    def take_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("take_id must not be empty")
        return value
