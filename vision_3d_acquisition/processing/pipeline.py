from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vision_3d_acquisition.calibration.calibration_models import SystemCalibration
from vision_3d_acquisition.calibration.calibration_storage import load_calibration
from vision_3d_acquisition.contracts.modalities import format_modalities, infer_modalities
from vision_3d_acquisition.contracts.execution import build_pipeline_execution_trace
from vision_3d_acquisition.contracts.result import ProcessingProfiling, ProcessingResult, ProcessingTiming
from vision_3d_acquisition.processing.calibrated_filtering import (
    evaluate_clusters_with_calibration,
    filter_by_labeled_planes,
)
from vision_3d_acquisition.processing.clustering import cluster_foreground
from vision_3d_acquisition.processing.measurements import measure_clusters
from vision_3d_acquisition.processing.measurements import measure_cluster
from vision_3d_acquisition.processing.plane_segmentation import segment_dominant_plane
from vision_3d_acquisition.processing.pointcloud_io import load_point_cloud, load_point_cloud_fast
from vision_3d_acquisition.processing.preprocessing import remove_outliers, voxel_downsample
from vision_3d_acquisition.processing.visualization import (
    render_belt_polygon_topview,
    render_calibrated_planes,
    render_clusters,
    render_filtered_clusters,
    render_foreground,
    render_plane_segmentation,
    render_point_cloud_preview,
    render_rejected_points,
)
from vision_3d_acquisition.utils.profiling import ProcessingProfiler
from vision_3d_acquisition.poc.calibration_mode import calibration_status
from vision_3d_acquisition.poc.summary import build_calibration_diagnostics, build_poc_run_summary
from vision_3d_acquisition.pipelines.registry import build_stage_outputs, default_pipeline_info
from vision_3d_acquisition.vision_core.serialization import write_result_json


@dataclass(frozen=True)
class ProcessingConfig:
    voxel_size: float = 1.0
    enable_outlier_removal: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    calibration_crop_margin_mm: float = 20.0
    plane_distance_threshold: float = 1.5
    plane_ransac_n: int = 3
    plane_num_iterations: int = 1000
    dbscan_eps: float = 3.0
    dbscan_min_points: int = 30
    min_cluster_points: int = 100


def load_processing_config(path: Path | None = None) -> ProcessingConfig:
    if path is None or not path.is_file():
        return ProcessingConfig()
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            if value.lower() in {"true", "false"}:
                values[key] = value.lower() == "true"
            else:
                values[key] = float(value) if "." in value else int(value)
        except ValueError:
            values[key] = value
    allowed = {field.name for field in ProcessingConfig.__dataclass_fields__.values()}
    return ProcessingConfig(**{key: value for key, value in values.items() if key in allowed})


def process_take(
    take_dir: Path,
    output_dir: Path,
    config: ProcessingConfig | None = None,
    calibration: SystemCalibration | str | Path | None = None,
    *,
    skip_debug_images: bool = False,
    prefer_fast_cloud: bool = True,
    calibration_resolution_source: str | None = None,
) -> ProcessingResult:
    config = config or ProcessingConfig()
    profiler = ProcessingProfiler()
    metadata_path = take_dir / "metadata.json"
    with profiler.stage("load_metadata", category="io") as stage:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stage.metadata["path"] = str(metadata_path)
    take_id = str(metadata["take_id"])
    session_id = metadata.get("session_id")
    frameset = metadata.get("frameset") if isinstance(metadata.get("frameset"), dict) else {}
    synchronization = (
        frameset.get("synchronization")
        if isinstance(frameset.get("synchronization"), dict)
        else {"mode": "none", "confidence": 1.0}
    )
    acquisition_timestamps = {
        "acquired_at": str(frameset.get("timestamp") or metadata.get("created_at")),
        "metadata_created_at": str(metadata.get("created_at")),
    }
    input_modalities = infer_modalities(metadata, take_dir)
    if "point_cloud" not in input_modalities:
        available = format_modalities(input_modalities)
        raise ValueError(f"Pipeline requires point_cloud, but take has {available}.")
    files_metadata = metadata.get("files", {})
    point_cloud_filename = files_metadata.get("point_cloud", "point_cloud.ply")
    point_cloud_npz_filename = files_metadata.get("point_cloud_npz")
    point_cloud_path = take_dir / point_cloud_filename
    fast_cloud_path = take_dir / point_cloud_npz_filename if point_cloud_npz_filename else None
    runtime_cloud_path = fast_cloud_path if prefer_fast_cloud and fast_cloud_path is not None and fast_cloud_path.is_file() else point_cloud_path
    load_stage_name = "load_point_cloud_fast" if runtime_cloud_path.suffix.lower() == ".npz" else "load_point_cloud"

    output_dir.mkdir(parents=True, exist_ok=True)
    with profiler.stage(load_stage_name, category="io", metadata={"path": str(runtime_cloud_path)}) as stage:
        original_cloud = load_point_cloud_fast(runtime_cloud_path) if load_stage_name == "load_point_cloud_fast" else load_point_cloud(runtime_cloud_path)
        stage.output_points = _point_count(original_cloud)
    with profiler.stage("compute_input_stats", category="preprocessing", input_points=_point_count(original_cloud)) as stage:
        input_stats = _compute_cloud_stats(original_cloud, runtime_cloud_path)
        stage.output_points = int(input_stats["point_count"])
    _render_or_skip(
        profiler,
        "debug_render_input_preview",
        "debug_artifacts",
        skip_debug_images,
        lambda: render_point_cloud_preview(original_cloud, output_dir / "input_point_cloud_preview.png"),
        input_points=_point_count(original_cloud),
    )
    with profiler.stage(
        "load_calibration",
        category="io",
        metadata={"path": str(calibration) if calibration is not None and not isinstance(calibration, SystemCalibration) else None},
    ) as stage:
        calibration_model, calibration_file = _load_optional_calibration(calibration)
        if calibration_model is not None and calibration_model.calibration_type != "plane_3d":
            raise ValueError(f"Unsupported calibration type: {calibration_model.calibration_type}")
        if calibration_model is not None and "point_cloud" not in input_modalities:
            raise ValueError(
                f"Calibration {calibration_model.calibration_id} is plane_3d and requires point_cloud "
                f"but take has {format_modalities(input_modalities)}"
            )
        stage.metadata["calibration_id"] = calibration_model.calibration_id if calibration_model is not None else None

    if calibration_model is not None:
        with profiler.stage("early_calibration_crop", category="calibration_filtering", input_points=_point_count(original_cloud)) as stage:
            processing_cloud = _early_calibration_crop(original_cloud, calibration_model, config.calibration_crop_margin_mm)
            stage.output_points = _point_count(processing_cloud)
            stage.metadata["margin_mm"] = config.calibration_crop_margin_mm
    else:
        processing_cloud = original_cloud
        profiler.record_skipped(
            "early_calibration_crop",
            category="calibration_filtering",
            notes="skipped: no calibration provided",
            input_points=_point_count(original_cloud),
            metadata={"margin_mm": config.calibration_crop_margin_mm},
        )

    with profiler.stage("voxel_downsample", category="preprocessing", input_points=_point_count(processing_cloud)) as stage:
        downsampled = voxel_downsample(processing_cloud, config.voxel_size)
        stage.output_points = _point_count(downsampled)
        stage.metadata["voxel_size"] = config.voxel_size
    with profiler.stage("remove_outliers", category="preprocessing", input_points=_point_count(downsampled)) as stage:
        if config.enable_outlier_removal:
            preprocessed = remove_outliers(
                downsampled,
                nb_neighbors=config.outlier_nb_neighbors,
                std_ratio=config.outlier_std_ratio,
            )
        else:
            preprocessed = downsampled
            stage.notes = "skipped: enable_outlier_removal=false"
        stage.output_points = _point_count(preprocessed)
        stage.metadata["enabled"] = config.enable_outlier_removal
        stage.metadata["nb_neighbors"] = config.outlier_nb_neighbors
        stage.metadata["std_ratio"] = config.outlier_std_ratio

    if calibration_model is None:
        with profiler.stage("outer_plane_filtering", category="calibration_filtering", input_points=_point_count(preprocessed)) as stage:
            segmentation = segment_dominant_plane(
                preprocessed,
                distance_threshold=config.plane_distance_threshold,
                ransac_n=config.plane_ransac_n,
                num_iterations=config.plane_num_iterations,
            )
            stage.output_points = _point_count(segmentation["foreground_cloud"])
            stage.metadata["plane_points"] = _point_count(segmentation["plane_cloud"])
        foreground = segmentation["foreground_cloud"]
        profiler.record_skipped(
            "belt_polygon_filtering",
            category="calibration_filtering",
            notes="skipped: no calibration provided",
            input_points=_point_count(foreground),
        )
        profiler.record_skipped(
            "belt_height_filtering",
            category="calibration_filtering",
            notes="skipped: no calibration provided",
            input_points=_point_count(foreground),
        )
        with profiler.stage("dbscan_clustering", category="clustering", input_points=_point_count(foreground)) as stage:
            clusters = cluster_foreground(
                foreground,
                dbscan_eps=config.dbscan_eps,
                dbscan_min_points=config.dbscan_min_points,
                min_cluster_points=config.min_cluster_points,
            )
            stage.output_objects = len(clusters)
            stage.output_points = _cluster_point_count(clusters)
        _render_or_skip(
            profiler,
            "debug_render_calibrated_planes",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_plane_segmentation(
                segmentation["plane_cloud"],
                foreground,
                output_dir / "debug_plane_segmentation.png",
            ),
            input_points=_point_count(preprocessed),
        )
        _render_or_skip(
            profiler,
            "debug_render_filtered_foreground",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_foreground(foreground, output_dir / "debug_foreground.png"),
            input_points=_point_count(foreground),
        )
        _render_or_skip(
            profiler,
            "debug_render_clusters_filtered",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_clusters(clusters, output_dir / "debug_clusters.png"),
            input_points=_cluster_point_count(clusters),
            input_objects=len(clusters),
        )
        with profiler.stage("cluster_measurements", category="measurement", input_objects=len(clusters), input_points=_cluster_point_count(clusters)) as stage:
            objects = measure_clusters(clusters)
            stage.output_objects = len(objects)
        with profiler.stage("sphere_diameter_placeholder", category="measurement", input_objects=len(objects), output_objects=len(objects)) as stage:
            stage.notes = "placeholder: diameter/sphere fitting not implemented"
        rejected_objects: list[dict[str, object]] = []
        algorithm_stage = "segmentation"
        files = {
            "point_cloud": point_cloud_filename,
            "point_cloud_npz": point_cloud_npz_filename,
            "input_preview": None if skip_debug_images else "input_point_cloud_preview.png",
            "debug_plane_segmentation": None if skip_debug_images else "debug_plane_segmentation.png",
            "debug_foreground": None if skip_debug_images else "debug_foreground.png",
            "debug_clusters": None if skip_debug_images else "debug_clusters.png",
        }
        plane_filtering = None
    else:
        with profiler.stage("outer_plane_filtering", category="calibration_filtering", input_points=_point_count(preprocessed)) as stage:
            segmentation = filter_by_labeled_planes(
                preprocessed,
                calibration_model,
                distance_threshold_mm=config.plane_distance_threshold,
            )
            stats = segmentation["statistics"]
            stage.output_points = int(stats["candidate_foreground_points"]) + int(stats["rejected_points"])
            stage.metadata["ignored_plane_points"] = stats["ignored_plane_points"]
        foreground = segmentation["foreground_cloud"]
        with profiler.stage("belt_height_filtering", category="calibration_filtering", input_points=_point_count(preprocessed)) as stage:
            stage.output_points = _point_count(foreground)
            stage.metadata["min_height_above_belt_mm"] = calibration_model.object_filter.min_height_above_belt_mm
            stage.metadata["max_height_above_belt_mm"] = calibration_model.object_filter.max_height_above_belt_mm
            stage.notes = "included in calibrated plane filtering"
        with profiler.stage("dbscan_clustering", category="clustering", input_points=_point_count(foreground)) as stage:
            raw_clusters = cluster_foreground(
                foreground,
                dbscan_eps=config.dbscan_eps,
                dbscan_min_points=config.dbscan_min_points,
                min_cluster_points=config.min_cluster_points,
            )
            stage.output_objects = len(raw_clusters)
            stage.output_points = _cluster_point_count(raw_clusters)
        with profiler.stage("belt_polygon_filtering", category="calibration_filtering", input_objects=len(raw_clusters), input_points=_cluster_point_count(raw_clusters)) as stage:
            clusters, rejected_clusters = evaluate_clusters_with_calibration(
                raw_clusters,
                calibration_model,
                min_cluster_points=config.min_cluster_points,
            )
            stage.output_objects = len(clusters)
            stage.output_points = _cluster_point_count(clusters)
            stage.metadata["rejected_objects"] = len(rejected_clusters)
        _render_or_skip(
            profiler,
            "debug_render_calibrated_planes",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_calibrated_planes(
                segmentation["belt_reference_points"],
                segmentation["ignored_plane_points"],
                segmentation["candidate_foreground_points"],
                output_dir / "debug_calibrated_planes.png",
            ),
            input_points=_point_count(preprocessed),
        )
        _render_or_skip(
            profiler,
            "debug_render_belt_polygon_topview",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_belt_polygon_topview(
                segmentation["candidate_foreground_points"],
                segmentation["rejected_points"],
                calibration_model.belt_plane().roi_polygon_xy_mm,
                output_dir / "debug_belt_polygon_topview.png",
            ),
            input_points=_point_count(segmentation["candidate_foreground_points"]) + _point_count(segmentation["rejected_points"]),
        )
        _render_or_skip(
            profiler,
            "debug_render_filtered_foreground",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_foreground(segmentation["candidate_foreground_points"], output_dir / "debug_filtered_foreground.png"),
            input_points=_point_count(segmentation["candidate_foreground_points"]),
        )
        _render_or_skip(
            profiler,
            "debug_render_rejected_points",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_rejected_points(segmentation["rejected_points"], output_dir / "debug_rejected_points.png"),
            input_points=_point_count(segmentation["rejected_points"]),
        )
        _render_or_skip(
            profiler,
            "debug_render_clusters_filtered",
            "debug_artifacts",
            skip_debug_images,
            lambda: render_filtered_clusters(clusters, output_dir / "debug_clusters_filtered.png"),
            input_points=_cluster_point_count(clusters),
            input_objects=len(clusters),
        )
        with profiler.stage("cluster_measurements", category="measurement", input_objects=len(clusters) + len(rejected_clusters), input_points=_cluster_point_count(clusters) + _cluster_point_count(rejected_clusters)) as stage:
            objects = _measure_calibrated_clusters(clusters)
            rejected_objects = _measure_calibrated_clusters(rejected_clusters)
            stage.output_objects = len(objects) + len(rejected_objects)
        with profiler.stage("sphere_diameter_placeholder", category="measurement", input_objects=len(objects), output_objects=len(objects)) as stage:
            stage.notes = "placeholder: diameter/sphere fitting not implemented"
        plane_filtering = dict(segmentation["statistics"])
        plane_filtering["kept_objects"] = len(objects)
        plane_filtering["rejected_objects"] = len(rejected_objects)
        algorithm_stage = "calibrated_segmentation"
        files = {
            "point_cloud": point_cloud_filename,
            "point_cloud_npz": point_cloud_npz_filename,
            "input_preview": None if skip_debug_images else "input_point_cloud_preview.png",
            "debug_calibrated_planes": None if skip_debug_images else "debug_calibrated_planes.png",
            "debug_belt_polygon_topview": None if skip_debug_images else "debug_belt_polygon_topview.png",
            "debug_filtered_foreground": None if skip_debug_images else "debug_filtered_foreground.png",
            "debug_rejected_points": None if skip_debug_images else "debug_rejected_points.png",
            "debug_clusters_filtered": None if skip_debug_images else "debug_clusters_filtered.png",
            "debug_plane_segmentation": None if skip_debug_images else "debug_calibrated_planes.png",
            "debug_foreground": None if skip_debug_images else "debug_filtered_foreground.png",
            "debug_clusters": None if skip_debug_images else "debug_clusters_filtered.png",
        }

    pipeline_info = default_pipeline_info("3d_ball_inspection")
    result = ProcessingResult(
        take_id=take_id,
        session_id=session_id,
        frameset_id=frameset.get("frameset_id"),
        processed_at=datetime.now(UTC),
        processing_mode="real",
        processing_engine="legacy",
        plane_mode="calibrated" if calibration_model is not None else "auto",
        calibration_status="calibrated" if calibration_model is not None else calibration_status(None),
        calibration_resolution_source=calibration_resolution_source or ("explicit" if calibration is not None else "none"),
        calibration_active=calibration_model is not None,
        input_modalities=input_modalities,
        output_modalities=["point_cloud", "debug_image"],
        processing_pipeline={
            **pipeline_info,
            "name": "legacy_segmentation",
            "required_modalities": ["point_cloud"],
        },
        stage_outputs=build_stage_outputs(files),
        calibration={
            "calibration_type": calibration_model.calibration_type if calibration_model is not None else "plane_3d",
            "source_modalities": list(calibration_model.source_modalities) if calibration_model is not None else ["point_cloud"],
            "resolution_source": calibration_resolution_source or ("explicit" if calibration is not None else "none"),
            "file": str(calibration_file) if calibration_file is not None else None,
            "active": calibration_model is not None,
        },
        algorithm_stage=algorithm_stage,
        status="ok",
        summary={
            "object_count": len(objects),
            "ball_count": 0,
            "non_ball_count": 0,
            "decision": "review",
            "confidence": None,
        },
        plane_model=segmentation["plane_model"],
        input_stats=input_stats,
        objects=objects,
        files=files,
        timing_ms=_legacy_timing(profiler),
        profiling=ProcessingProfiling.model_validate(profiler.as_dict()),
        error=None,
        calibration_id=calibration_model.calibration_id if calibration_model is not None else None,
        calibration_file=str(calibration_file) if calibration_file is not None else None,
        calibration_snapshot=calibration_model.model_dump(mode="json") if calibration_model is not None else None,
        plane_filtering=plane_filtering,
        rejected_objects=rejected_objects,
        synchronization=synchronization,
        acquisition_timestamps=acquisition_timestamps,
    )
    result.calibration_diagnostics = build_calibration_diagnostics(result.model_dump(mode="json"), metadata=metadata)
    with profiler.stage("result_json_write", category="output") as stage:
        write_result_json(output_dir / "result.json", result.model_dump(mode="json"))
        stage.metadata["path"] = str(output_dir / "result.json")
    profiler.finish()
    result.profiling = ProcessingProfiling.model_validate(profiler.as_dict())
    result.timing_ms = _legacy_timing(profiler)
    result.calibration_diagnostics = build_calibration_diagnostics(result.model_dump(mode="json"), metadata=metadata)
    result.poc_summary = build_poc_run_summary(
        result.model_dump(mode="json"),
        metadata=metadata,
        output_dir=output_dir,
        engine="legacy",
    )
    result.pipeline_execution = build_pipeline_execution_trace(
        pipeline=result.processing_pipeline.model_dump(mode="json") if result.processing_pipeline is not None else None,
        artifacts=[item.model_dump(mode="json") for item in result.artifacts],
        profiling=result.profiling.model_dump(mode="json") if result.profiling is not None else None,
        input_modalities=list(result.input_modalities),
        objects=[item.model_dump(mode="json") for item in result.objects],
        rejected_objects=[item.model_dump(mode="json") for item in result.rejected_objects],
        result_status=result.status,
        result_error=result.error,
    )
    write_result_json(output_dir / "result.json", result.model_dump(mode="json"))
    return result


def _load_optional_calibration(calibration: SystemCalibration | str | Path | None) -> tuple[SystemCalibration | None, Path | None]:
    if calibration is None:
        return None, None
    if isinstance(calibration, SystemCalibration):
        return calibration, None
    path = Path(calibration)
    return load_calibration(path), path


def _measure_calibrated_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for cluster in clusters:
        measured = measure_cluster(int(cluster["cluster_id"]), cluster["cloud"])
        metrics = cluster.get("calibration_metrics", {})
        measured.update(
            {
                "height_above_belt_mm": metrics.get("height_above_belt_mm"),
                "fraction_points_inside_belt": metrics.get("fraction_points_inside_belt"),
                "filter_status": metrics.get("filter_status"),
                "filter_reason": metrics.get("filter_reason"),
            }
        )
        objects.append(measured)
    return objects


def _compute_cloud_stats(cloud: Any, source_path: Path) -> dict[str, Any]:
    points = np.asarray(cloud.points)
    if points.size == 0:
        min_bound = max_bound = extent = np.zeros(3, dtype=float)
    else:
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        extent = max_bound - min_bound
    return {
        "point_count": int(points.shape[0]),
        "has_colors": bool(cloud.has_colors()),
        "has_normals": bool(cloud.has_normals()),
        "min_bound": _round_vector(min_bound),
        "max_bound": _round_vector(max_bound),
        "extent": _round_vector(extent),
        "file_size_bytes": source_path.stat().st_size if source_path.is_file() else None,
    }


def _early_calibration_crop(cloud: Any, calibration: SystemCalibration, margin_mm: float) -> Any:
    points = np.asarray(cloud.points)
    if len(points) == 0:
        return cloud.select_by_index([])
    belt = calibration.belt_plane()
    if len(belt.roi_polygon_xy_mm) >= 3:
        xy = np.asarray(belt.roi_polygon_xy_mm, dtype=float)
        xy_min = xy.min(axis=0) - margin_mm
        xy_max = xy.max(axis=0) + margin_mm
    else:
        bbox_min = np.asarray(belt.bbox_min_mm, dtype=float)
        bbox_max = np.asarray(belt.bbox_max_mm, dtype=float)
        xy_min = bbox_min[:2] - margin_mm
        xy_max = bbox_max[:2] + margin_mm
    bbox_min = np.asarray(belt.bbox_min_mm, dtype=float)
    bbox_max = np.asarray(belt.bbox_max_mm, dtype=float)
    z_min = min(float(bbox_min[2]), float(bbox_max[2])) - margin_mm
    z_max = max(float(bbox_min[2]), float(bbox_max[2])) + calibration.object_filter.max_height_above_belt_mm + margin_mm
    mask = (
        (points[:, 0] >= xy_min[0])
        & (points[:, 0] <= xy_max[0])
        & (points[:, 1] >= xy_min[1])
        & (points[:, 1] <= xy_max[1])
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return cloud.select_by_index(np.where(mask)[0].tolist())


def _point_count(cloud: Any) -> int:
    return len(cloud.points)


def _cluster_point_count(clusters: list[dict[str, Any]]) -> int:
    return sum(_point_count(cluster["cloud"]) for cluster in clusters)


def _render_or_skip(
    profiler: ProcessingProfiler,
    name: str,
    category: str,
    skip_debug_images: bool,
    render: Any,
    *,
    input_points: int | None = None,
    input_objects: int | None = None,
) -> None:
    if skip_debug_images:
        profiler.record_skipped(
            name,
            category=category,
            notes="skipped by --skip-debug-images",
            input_points=input_points,
            input_objects=input_objects,
        )
        return
    with profiler.stage(name, category=category, input_points=input_points, input_objects=input_objects):
        render()


def _legacy_timing(profiler: ProcessingProfiler) -> ProcessingTiming:
    io_ms = profiler.aggregate_ms("io")
    classification_ms = profiler.aggregate_ms("classification")
    production_without_io_classification = sum(
        stage.duration_ms
        for stage in profiler.stages
        if stage.category not in {"io", "classification", "debug_artifacts", "output"}
    )
    return ProcessingTiming(
        load=max(0, int(round(io_ms))),
        segmentation=max(0, int(round(production_without_io_classification))),
        classification=max(0, int(round(classification_ms))),
        total=max(0, int(round(profiler.total_ms))),
    )


def _round_vector(vector: np.ndarray) -> tuple[float, float, float]:
    values = [round(float(value), 4) for value in vector.tolist()]
    return (values[0], values[1], values[2])
