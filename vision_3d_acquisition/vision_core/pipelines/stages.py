from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision_3d_acquisition.calibration.calibration_models import SystemCalibration
from vision_3d_acquisition.calibration.calibration_storage import load_calibration
from vision_3d_acquisition.contracts.artifacts import normalize_processing_artifacts
from vision_3d_acquisition.contracts.execution import build_pipeline_execution_trace
from vision_3d_acquisition.contracts.modalities import CaptureModality
from vision_3d_acquisition.contracts.result import ProcessingProfiling, ProcessingResult, ProcessingTiming
from vision_3d_acquisition.processing.calibrated_filtering import (
    evaluate_clusters_with_calibration,
    filter_by_labeled_planes,
)
from vision_3d_acquisition.processing.clustering import cluster_foreground
from vision_3d_acquisition.processing.measurements import measure_cluster, measure_clusters
from vision_3d_acquisition.processing.pipeline import ProcessingConfig, process_take
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
from vision_3d_acquisition.pipelines.registry import build_stage_outputs_from_artifacts, default_pipeline_info
from vision_3d_acquisition.vision_core.acquisition.sources import AcquisitionSource
from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext
from vision_3d_acquisition.vision_core.projections import generate_canonical_projections, projection_metadata_payload


@dataclass
class LoadCaptureStage:
    source: AcquisitionSource
    take_id: str | None = None
    output_root: Path = Path("data/processed")
    name: str = "LoadCaptureStage"
    category: str = "io"
    produced_artifacts: tuple[str, ...] = ("capture_ref", "metadata", "modalities", "assets")
    produced_modalities: tuple[CaptureModality, ...] = ()
    required_modalities: tuple[CaptureModality, ...] = ()
    stage_category: str | None = "io"

    def run(self, context: PipelineContext) -> None:
        reference = self.source.by_id(self.take_id) if self.take_id else self.source.latest()
        output_dir = self.output_root / reference.take_id
        output_dir.mkdir(parents=True, exist_ok=True)
        context.set_artifact("capture_ref", reference)
        context.set_artifact("take_id", reference.take_id)
        context.set_artifact("take_dir", reference.take_dir)
        context.set_artifact("metadata", reference.metadata)
        context.set_artifact("modalities", reference.modalities)
        context.set_artifact("assets", reference.assets)
        context.set_artifact("frame_count", reference.frame_count)
        context.set_artifact("output_dir", output_dir)


@dataclass
class DecodeHeightmapStage:
    prefer_fast_cloud: bool = True
    render_preview: bool = True
    name: str = "DecodeHeightmapStage"
    category: str = "preprocessing"
    required_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_artifacts: tuple[str, ...] = ("cloud", "input_stats")
    stage_category: str | None = "preprocessing"

    def run(self, context: PipelineContext) -> None:
        metadata = context.require_artifact("metadata")
        take_dir: Path = context.require_artifact("take_dir")
        output_dir: Path = context.require_artifact("output_dir")
        runtime_cloud_path, point_cloud_filename, point_cloud_npz = _resolve_runtime_cloud_path(
            take_dir, metadata, self.prefer_fast_cloud
        )
        cloud = (
            load_point_cloud_fast(runtime_cloud_path)
            if runtime_cloud_path.suffix.lower() == ".npz"
            else load_point_cloud(runtime_cloud_path)
        )
        context.set_artifact("runtime_cloud_path", runtime_cloud_path)
        context.set_artifact("cloud", cloud)
        context.set_artifact("files.point_cloud", point_cloud_filename)
        context.set_artifact("files.point_cloud_npz", point_cloud_npz)
        context.set_artifact("input_stats", _compute_cloud_stats(cloud, runtime_cloud_path))
        if self.render_preview:
            preview_path = output_dir / "input_point_cloud_preview.png"
            render_point_cloud_preview(cloud, preview_path)
            context.add_debug_output("point_cloud_preview", preview_path)


@dataclass
class ApplyCalibrationStage:
    calibration_path: Path | None = None
    name: str = "ApplyCalibrationStage"
    category: str = "calibration_filtering"
    required_modalities: tuple[CaptureModality, ...] = ()
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("calibration",)
    stage_category: str | None = "calibration"

    def run(self, context: PipelineContext) -> None:
        calibration: SystemCalibration | None = None
        if self.calibration_path is not None:
            calibration = load_calibration(self.calibration_path)
            _validate_calibration_modalities(calibration, context)
            context.set_artifact("calibration_id", calibration.calibration_id)
        context.set_artifact("calibration", calibration)
        context.set_artifact(
            "calibration_path",
            str(self.calibration_path) if self.calibration_path is not None else None,
        )
        context.set_artifact("calibration_resolution_source", context.get_artifact("calibration_resolution_source", "explicit" if self.calibration_path else "none"))


@dataclass
class PreprocessPointCloudStage:
    voxel_size: float = 1.0
    enable_outlier_removal: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    calibration_crop_margin_mm: float = 20.0
    name: str = "PreprocessPointCloudStage"
    category: str = "preprocessing"
    required_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_artifacts: tuple[str, ...] = ("preprocessed_cloud",)
    stage_category: str | None = "preprocessing"

    def run(self, context: PipelineContext) -> None:
        cloud = context.require_artifact("cloud")
        calibration: SystemCalibration | None = context.get_artifact("calibration")
        processing_cloud = (
            _early_calibration_crop(cloud, calibration, self.calibration_crop_margin_mm)
            if calibration is not None
            else cloud
        )
        downsampled = voxel_downsample(processing_cloud, self.voxel_size)
        preprocessed = downsampled
        if self.enable_outlier_removal:
            preprocessed = remove_outliers(
                downsampled,
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio,
            )
        context.set_artifact("preprocessed_cloud", preprocessed)


@dataclass
class PlaneFilterStage:
    distance_threshold: float = 1.5
    ransac_n: int = 3
    num_iterations: int = 1000
    name: str = "PlaneFilterStage"
    category: str = "calibration_filtering"
    required_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_artifacts: tuple[str, ...] = ("foreground_cloud", "plane_model")
    stage_category: str | None = "calibration_filtering"

    def run(self, context: PipelineContext) -> None:
        cloud = context.require_artifact("preprocessed_cloud")
        calibration: SystemCalibration | None = context.get_artifact("calibration")
        if calibration is None:
            segmentation = segment_dominant_plane(
                cloud,
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations,
            )
            context.set_artifact("foreground_cloud", segmentation["foreground_cloud"])
            context.set_artifact("plane_filtering", None)
            context.set_artifact("segmentation_mode", "generic")
        else:
            segmentation = filter_by_labeled_planes(
                cloud,
                calibration,
                distance_threshold_mm=self.distance_threshold,
            )
            context.set_artifact("foreground_cloud", segmentation["foreground_cloud"])
            context.set_artifact("segmentation_mode", "calibrated")
            context.set_artifact("plane_filtering", dict(segmentation["statistics"]))
        context.set_artifact("segmentation", segmentation)
        context.set_artifact("plane_model", segmentation["plane_model"])


@dataclass
class SegmentObjectsStage:
    dbscan_eps: float = 3.0
    dbscan_min_points: int = 30
    min_cluster_points: int = 100
    name: str = "SegmentObjectsStage"
    category: str = "clustering"
    required_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("clusters",)
    stage_category: str | None = "clustering"

    def run(self, context: PipelineContext) -> None:
        foreground = context.require_artifact("foreground_cloud")
        calibration: SystemCalibration | None = context.get_artifact("calibration")
        clusters = cluster_foreground(
            foreground,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_points=self.dbscan_min_points,
            min_cluster_points=self.min_cluster_points,
        )
        rejected_clusters: list[dict[str, Any]] = []
        if calibration is not None:
            clusters, rejected_clusters = evaluate_clusters_with_calibration(
                clusters,
                calibration,
                min_cluster_points=self.min_cluster_points,
            )
            plane_filtering = dict(context.get_artifact("plane_filtering") or {})
            plane_filtering["kept_objects"] = len(clusters)
            plane_filtering["rejected_objects"] = len(rejected_clusters)
            context.set_artifact("plane_filtering", plane_filtering)
        context.set_artifact("clusters", clusters)
        context.set_artifact("rejected_clusters", rejected_clusters)
        context.metrics["cluster_count"] = len(clusters)


@dataclass
class MeasureObjectsStage:
    name: str = "MeasureObjectsStage"
    category: str = "measurement"
    required_modalities: tuple[CaptureModality, ...] = ()
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("objects",)
    stage_category: str | None = "measurement"

    def run(self, context: PipelineContext) -> None:
        clusters = context.require_artifact("clusters")
        rejected_clusters = context.get_artifact("rejected_clusters", [])
        objects = measure_clusters(clusters)
        rejected_objects = _measure_calibrated_clusters(rejected_clusters)
        context.set_artifact("objects", objects)
        context.set_artifact("rejected_objects", rejected_objects)
        context.metrics["object_count"] = len(objects)
        context.add_processing_artifact(
            artifact_id="measurement_table",
            stage_id="measurement",
            kind="table",
            title="Object measurements",
            metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
        )
        context.add_processing_artifact(
            artifact_id="measurement_summary",
            stage_id="measurement",
            kind="metric",
            title="Diameter/statistics summary",
            metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
        )


@dataclass
class RenderDebugArtifactsStage:
    skip_debug_images: bool = False
    name: str = "RenderDebugArtifactsStage"
    category: str = "debug_artifacts"
    required_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("files", "debug_images")
    stage_category: str | None = "debug_artifacts"

    def run(self, context: PipelineContext) -> None:
        output_dir: Path = context.require_artifact("output_dir")
        point_cloud_filename = context.get_artifact("files.point_cloud")
        point_cloud_npz = context.get_artifact("files.point_cloud_npz")
        segmentation = context.require_artifact("segmentation")
        clusters = context.require_artifact("clusters")
        calibration: SystemCalibration | None = context.get_artifact("calibration")
        foreground = segmentation.get("foreground_cloud")

        files: dict[str, str | None] = {
            "point_cloud": point_cloud_filename,
            "point_cloud_npz": point_cloud_npz,
            "input_preview": None if self.skip_debug_images else "input_point_cloud_preview.png",
        }
        if self.skip_debug_images:
            if calibration is None:
                files.update(
                    {
                        "debug_plane_segmentation": None,
                        "debug_foreground": None,
                        "debug_clusters": None,
                    }
                )
            else:
                files.update(
                    {
                        "debug_calibrated_planes": None,
                        "debug_belt_polygon_topview": None,
                        "debug_filtered_foreground": None,
                        "debug_rejected_points": None,
                        "debug_clusters_filtered": None,
                        "debug_plane_segmentation": None,
                        "debug_foreground": None,
                        "debug_clusters": None,
                    }
                )
            context.set_artifact("files", files)
            return

        if calibration is None:
            render_plane_segmentation(
                segmentation["plane_cloud"],
                segmentation["foreground_cloud"],
                output_dir / "debug_plane_segmentation.png",
            )
            render_foreground(segmentation["foreground_cloud"], output_dir / "debug_foreground.png")
            render_clusters(clusters, output_dir / "debug_clusters.png")
            files.update(
                {
                    "debug_plane_segmentation": "debug_plane_segmentation.png",
                    "debug_foreground": "debug_foreground.png",
                    "debug_clusters": "debug_clusters.png",
                }
            )
            context.add_processing_artifact(
                artifact_id="plane_segmentation",
                stage_id="segmentation",
                kind="image",
                title="Plane segmentation",
                path="debug_plane_segmentation.png",
                mime_type="image/png",
                preview_available=True,
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
            )
            context.add_processing_artifact(
                artifact_id="foreground_extraction",
                stage_id="segmentation",
                kind="image",
                title="Foreground extraction",
                path="debug_foreground.png",
                mime_type="image/png",
                preview_available=True,
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
            )
            context.add_processing_artifact(
                artifact_id="foreground_clusters",
                stage_id="segmentation",
                kind="image",
                title="Foreground clusters",
                path="debug_clusters.png",
                mime_type="image/png",
                preview_available=True,
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
            )
        else:
            rejected_cloud = segmentation["rejected_points"]
            render_calibrated_planes(
                segmentation["belt_reference_points"],
                segmentation["ignored_plane_points"],
                segmentation["candidate_foreground_points"],
                output_dir / "debug_calibrated_planes.png",
            )
            render_belt_polygon_topview(
                segmentation["candidate_foreground_points"],
                rejected_cloud,
                calibration.belt_plane().roi_polygon_xy_mm,
                output_dir / "debug_belt_polygon_topview.png",
            )
            render_foreground(segmentation["candidate_foreground_points"], output_dir / "debug_filtered_foreground.png")
            render_rejected_points(rejected_cloud, output_dir / "debug_rejected_points.png")
            render_filtered_clusters(clusters, output_dir / "debug_clusters_filtered.png")
            files.update(
                {
                    "debug_calibrated_planes": "debug_calibrated_planes.png",
                    "debug_belt_polygon_topview": "debug_belt_polygon_topview.png",
                    "debug_filtered_foreground": "debug_filtered_foreground.png",
                    "debug_rejected_points": "debug_rejected_points.png",
                    "debug_clusters_filtered": "debug_clusters_filtered.png",
                    "debug_plane_segmentation": "debug_calibrated_planes.png",
                    "debug_foreground": "debug_filtered_foreground.png",
                    "debug_clusters": "debug_clusters_filtered.png",
                }
            )
            context.add_processing_artifact(
                artifact_id="calibrated_planes",
                stage_id="segmentation",
                kind="image",
                title="Calibrated planes",
                path="debug_calibrated_planes.png",
                mime_type="image/png",
                preview_available=True,
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
            )
            context.add_processing_artifact(
                artifact_id="foreground_extraction",
                stage_id="segmentation",
                kind="image",
                title="Foreground extraction",
                path="debug_filtered_foreground.png",
                mime_type="image/png",
                preview_available=True,
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
            )
            context.add_processing_artifact(
                artifact_id="foreground_clusters",
                stage_id="segmentation",
                kind="image",
                title="Foreground clusters",
                path="debug_clusters_filtered.png",
                mime_type="image/png",
                preview_available=True,
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
            )
        context.set_artifact("files", files)
        if foreground is not None:
            points = np.asarray(foreground.points)
            for projection in generate_canonical_projections(points=points, output_dir=output_dir, prefix="projection"):
                files[f"projection_{projection.projection_type}"] = projection.filename
                metadata = projection_metadata_payload(projection)
                context.add_processing_artifact(
                    artifact_id=projection.projection_type,
                    stage_id="segmentation",
                    kind="image",
                    title=projection.projection_type.replace("_", " ").title(),
                    path=projection.filename,
                    mime_type="image/png",
                    preview_available=True,
                    projection_type=projection.projection_type,
                    projection_coordinate_system=metadata["coordinate_system"],
                    projection_metadata=metadata,
                    projection_transform_id=metadata["transform_id"],
                    metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
                )
        context.set_artifact("files", files)


@dataclass
class SerializeProcessingResultStage:
    name: str = "SerializeProcessingResultStage"
    category: str = "output"
    required_modalities: tuple[CaptureModality, ...] = ()
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("result_json",)
    stage_category: str | None = "output"

    def run(self, context: PipelineContext) -> None:
        take_id = context.require_artifact("take_id")
        metadata = context.get_artifact("metadata", {})
        frameset = metadata.get("frameset") if isinstance(metadata, dict) and isinstance(metadata.get("frameset"), dict) else {}
        synchronization = (
            frameset.get("synchronization")
            if isinstance(frameset.get("synchronization"), dict)
            else {"mode": "none", "confidence": 1.0}
        )
        calibration: SystemCalibration | None = context.get_artifact("calibration")
        calibration_path = context.get_artifact("calibration_path")
        objects = context.get_artifact("objects", [])
        rejected_objects = context.get_artifact("rejected_objects", [])
        summary = context.get_artifact(
            "summary",
            {
                "object_count": len(objects),
                "ball_count": 0,
                "non_ball_count": 0,
                "decision": "review",
                "confidence": None,
            },
        )
        algorithm_stage = context.get_artifact(
            "algorithm_stage",
            "calibrated_segmentation" if calibration is not None else "segmentation",
        )
        profiling = ProcessingProfiling.model_validate(context.profiler.as_dict())
        input_modalities = list(context.get_artifact("modalities", ["point_cloud"]))
        files = context.get_artifact("files", {})
        explicit_artifacts = list(context.processing_artifacts)
        explicit_artifacts.extend(
            [
                {
                    "artifact_id": "classification_table",
                    "stage_id": "classification",
                    "kind": "table",
                    "title": "Classification results",
                    "preview_available": False,
                    "metadata": {"source": "native_pipeline", "producer": "BallClassificationStage", "modality": "point_cloud"},
                    "generated": "explicit",
                },
                {
                    "artifact_id": "result_payload",
                    "stage_id": "result",
                    "kind": "json",
                    "title": "Full result payload",
                    "path": "result.json",
                    "mime_type": "application/json",
                    "preview_available": True,
                    "metadata": {"source": "native_pipeline", "producer": self.name},
                    "generated": "explicit",
                },
            ]
        )
        pipeline_info = default_pipeline_info()
        result = ProcessingResult(
            take_id=take_id,
            session_id=metadata.get("session_id") if isinstance(metadata, dict) else None,
            frameset_id=frameset.get("frameset_id"),
            processed_at=_utc_now(),
            processing_mode="real",
            processing_engine=None,
            plane_mode="calibrated" if calibration is not None else "auto",
            calibration_status="calibrated" if calibration is not None else "automatic plane estimation only",
            calibration_resolution_source=context.get_artifact("calibration_resolution_source"),
            calibration_active=calibration is not None,
            input_modalities=input_modalities,
            output_modalities=["point_cloud", "debug_image"],
            processing_pipeline={
                **pipeline_info,
                "name": context.get_artifact("pipeline_name", "inspection"),
                "required_modalities": list(context.get_artifact("pipeline_required_modalities", ["point_cloud"])),
            },
            stage_outputs=[],
            artifacts=[],
            calibration={
                "calibration_type": calibration.calibration_type if calibration is not None else "plane_3d",
                "source_modalities": list(calibration.source_modalities) if calibration is not None else ["point_cloud"],
                "resolution_source": context.get_artifact("calibration_resolution_source"),
                "file": calibration_path if calibration is not None else None,
                "active": calibration is not None,
            },
            algorithm_stage=algorithm_stage,
            status="ok",
            summary=summary,
            plane_model=context.get_artifact("plane_model"),
            input_stats=context.get_artifact("input_stats"),
            objects=objects,
            files=files,
            timing_ms=_legacy_timing_from_profiling(profiling),
            profiling=profiling,
            error=None,
            calibration_id=calibration.calibration_id if calibration is not None else None,
            calibration_file=calibration_path if calibration is not None else None,
            calibration_snapshot=calibration.model_dump(mode="json") if calibration is not None else None,
            plane_filtering=context.get_artifact("plane_filtering"),
            rejected_objects=rejected_objects,
            synchronization=synchronization,
            acquisition_timestamps={
                "acquired_at": str(frameset.get("timestamp") or (metadata or {}).get("created_at")),
                "metadata_created_at": str((metadata or {}).get("created_at")),
            },
        )
        result_payload = result.model_dump(mode="json")
        result_payload["artifacts"] = normalize_processing_artifacts(result_payload | {"artifacts": explicit_artifacts}, output_dir=context.get_artifact("output_dir"))
        result_payload["stage_outputs"] = build_stage_outputs_from_artifacts(result_payload["artifacts"])
        result_payload["pipeline_execution"] = build_pipeline_execution_trace(
            pipeline=result_payload.get("processing_pipeline"),
            artifacts=result_payload["artifacts"],
            profiling=result_payload.get("profiling"),
            input_modalities=result_payload.get("input_modalities"),
            objects=result_payload.get("objects") or [],
            rejected_objects=result_payload.get("rejected_objects") or [],
            result_status=result_payload.get("status"),
            result_error=result_payload.get("error"),
        )
        for item in result_payload.get("objects", []):
            object_id = item.get("object_id")
            if isinstance(object_id, int):
                item["artifact_ids"] = [
                    artifact["artifact_id"]
                    for artifact in result_payload["artifacts"]
                    if artifact.get("object_id") == object_id
                ]
        result = ProcessingResult.model_validate(result_payload)
        context.set_artifact("processing_result", result)


@dataclass
class RunLegacySegmentationStage:
    config: ProcessingConfig | None = None
    calibration_path: Path | None = None
    skip_debug_images: bool = False
    prefer_fast_cloud: bool = True
    name: str = "RunLegacySegmentationStage"
    category: str = "classification"
    required_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_modalities: tuple[CaptureModality, ...] = ("point_cloud",)
    produced_artifacts: tuple[str, ...] = ("processing_result",)
    stage_category: str | None = "classification"

    def run(self, context: PipelineContext) -> None:
        take_dir: Path = context.require_artifact("take_dir")
        output_dir: Path = context.require_artifact("output_dir")
        result = process_take(
            take_dir,
            output_dir,
            config=self.config or ProcessingConfig(),
            calibration=self.calibration_path,
            skip_debug_images=self.skip_debug_images,
            prefer_fast_cloud=self.prefer_fast_cloud,
        )
        context.set_artifact("processing_result", result)
        context.set_artifact("result_path", output_dir / "result.json")
        context.metrics["object_count"] = len(result.objects)


def _resolve_runtime_cloud_path(
    take_dir: Path, metadata: dict[str, Any], prefer_fast_cloud: bool
) -> tuple[Path, str, str | None]:
    files = metadata.get("files", {})
    point_cloud_filename = str(files.get("point_cloud", "point_cloud.ply"))
    point_cloud_npz = files.get("point_cloud_npz")
    point_cloud_path = take_dir / point_cloud_filename
    fast_cloud_path = take_dir / str(point_cloud_npz) if point_cloud_npz else None
    runtime_cloud_path = (
        fast_cloud_path
        if prefer_fast_cloud and fast_cloud_path is not None and fast_cloud_path.is_file()
        else point_cloud_path
    )
    return runtime_cloud_path, point_cloud_filename, str(point_cloud_npz) if point_cloud_npz else None


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


def _round_vector(vector: np.ndarray) -> tuple[float, float, float]:
    values = [round(float(value), 4) for value in vector.tolist()]
    return (values[0], values[1], values[2])


def _validate_calibration_modalities(calibration: SystemCalibration, context: PipelineContext) -> None:
    if calibration.calibration_type != "plane_3d":
        raise ValueError(f"Unsupported calibration type: {calibration.calibration_type}")
    available = set(context.get_artifact("modalities", []))
    if "point_cloud" not in available:
        raise ValueError(
            f"Calibration {calibration.calibration_id} is plane_3d and requires point_cloud "
            f"but take has {', '.join(sorted(available)) or 'none'}"
        )


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


def _legacy_timing_from_profiling(profiling: ProcessingProfiling) -> ProcessingTiming:
    io_ms = sum(stage.duration_ms for stage in profiling.stages if stage.category == "io")
    classification_ms = sum(stage.duration_ms for stage in profiling.stages if stage.category == "classification")
    production_without_io_classification = sum(
        stage.duration_ms
        for stage in profiling.stages
        if stage.category not in {"io", "classification", "debug_artifacts", "output"}
    )
    return ProcessingTiming(
        load=max(0, int(round(io_ms))),
        segmentation=max(0, int(round(production_without_io_classification))),
        classification=max(0, int(round(classification_ms))),
        total=max(0, int(round(profiling.total_ms))),
    )


def _utc_now():
    from datetime import UTC, datetime

    return datetime.now(UTC)
