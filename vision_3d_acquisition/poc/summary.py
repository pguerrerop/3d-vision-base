from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.contracts.result import ProcessingResult
from vision_3d_acquisition.poc.calibration_mode import calibration_display, calibration_status


LOW_POINT_COUNT = 100
HIGH_REJECTION_RATE = 0.5
ABNORMAL_TOTAL_MS = 5000.0
MAX_PLANE_TILT_DEG = 12.0


def validate_result_payload(payload: dict[str, Any]) -> ProcessingResult:
    return ProcessingResult.model_validate(payload)


def build_poc_run_summary(
    result_payload: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    result = validate_result_payload(result_payload)
    metadata = metadata or {}
    profiling = result.profiling
    input_stats = result.input_stats
    plane_filtering = result.plane_filtering or {}
    objects = list(result.objects)
    rejected_objects = list(result.rejected_objects)
    all_objects = objects + rejected_objects
    calibration_diagnostics = build_calibration_diagnostics(result_payload, metadata=metadata)
    warnings = _build_warnings(result, calibration_diagnostics)
    debug_images = _debug_images(result_payload, output_dir)
    point_cloud_present = _artifact_present(result.files.point_cloud, output_dir)
    result_json_present = (output_dir / "result.json").is_file() if output_dir is not None else True
    slowest_stage = _slowest_stage(result_payload)
    confidence_values = [item.confidence for item in all_objects if item.confidence is not None]
    fit_scores = [_fit_score(item.model_dump(mode="json")) for item in all_objects]
    fit_scores = [score for score in fit_scores if score is not None]
    rejected_count = len(rejected_objects)
    accepted_count = sum(1 for item in objects if item.filter_status != "rejected")
    plane_inlier_percent = _plane_inlier_percent(result_payload)
    built_engine = engine or result.processing_engine or _infer_engine(result_payload)
    return {
        "take_id": result.take_id,
        "timestamp": result.processed_at.isoformat(),
        "engine": built_engine,
        "calibration": {
            "id": result.calibration_id,
            "path": result.calibration_file,
            "display": calibration_display(result.calibration_file),
            "mode": result.plane_mode or ("calibrated" if result.calibration_file else "auto"),
            "status": result.calibration_status or calibration_status(result.calibration_file),
            "active": bool(result.calibration_active),
            "resolution_source": result.calibration_resolution_source or ("explicit" if result.calibration_file else "none"),
            "valid": not calibration_diagnostics["warnings"],
            "diagnostics": calibration_diagnostics,
        },
        "status": {
            "processing_status": result.status,
            "demo_ready": result.status == "ok" and not warnings,
            "trustworthy": result.status == "ok" and len(warnings) == 0,
            "calibration_valid": not calibration_diagnostics["warnings"],
        },
        "input": {
            "point_count": input_stats.point_count if input_stats else None,
            "valid_point_percent": 100.0 if input_stats and input_stats.point_count > 0 else 0.0,
            "dimensions_mm": list(input_stats.extent) if input_stats else None,
            "encoder_used": bool(metadata.get("encoder") or metadata.get("encoder_ticks")),
            "capture_source": metadata.get("source") or metadata.get("mode"),
        },
        "plane": {
            "found": result.plane_model is not None,
            "inlier_percent": plane_inlier_percent,
            "angle_deg": _plane_angle_deg(result.plane_model),
            "filtering_enabled": bool(result.calibration_id or plane_filtering),
        },
        "objects": {
            "found": len(objects),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "estimated_balls": result.summary.ball_count,
            "estimated_non_balls": result.summary.non_ball_count,
        },
        "classification": {
            "average_fit_score": _average(fit_scores),
            "worst_fit_score": min(fit_scores) if fit_scores else None,
            "confidence_average": _average(confidence_values),
            "confidence_min": min(confidence_values) if confidence_values else None,
            "confidence_max": max(confidence_values) if confidence_values else None,
        },
        "profiling": {
            "slowest_stage": slowest_stage,
            "total_processing_ms": profiling.total_ms if profiling else float(result.timing_ms.total),
            "bottleneck_category": slowest_stage["category"] if slowest_stage else None,
        },
        "artifacts": {
            "debug_images_present": debug_images,
            "point_cloud_present": point_cloud_present,
            "overlays_present": _artifact_present(result.files.overlay, output_dir),
            "result_json_present": result_json_present,
        },
        "warnings": warnings,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def build_calibration_diagnostics(result_payload: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result = validate_result_payload(result_payload)
    metadata = metadata or {}
    angle = _plane_angle_deg(result.plane_model)
    input_stats = result.input_stats
    plane_inlier_percent = _plane_inlier_percent(result_payload)
    warnings: list[str] = []
    if result.calibration_id and angle is not None and angle > MAX_PLANE_TILT_DEG:
        warnings.append("excessive_plane_tilt")
    if result.calibration_id and plane_inlier_percent is not None and plane_inlier_percent < 20.0:
        warnings.append("low_plane_inlier_percent")
    if input_stats is not None:
        extent = list(input_stats.extent)
        max_extent = max(extent) if extent else 0.0
        min_extent = min(value for value in extent if value > 0.0) if any(value > 0.0 for value in extent) else 0.0
        if max_extent > 0 and min_extent > 0 and max_extent / min_extent > 100.0:
            warnings.append("suspicious_scaling")
        if input_stats.point_count < LOW_POINT_COUNT:
            warnings.append("abnormal_point_density")
    if result.plane_filtering:
        rejected_points = float(result.plane_filtering.get("rejected_points") or 0)
        input_points = float(result.plane_filtering.get("input_points") or 0)
        if input_points > 0 and rejected_points / input_points > 0.8:
            warnings.append("unusual_offsets")
    if result.calibration_id and not metadata.get("encoder") and not metadata.get("encoder_ticks"):
        warnings.append("encoder_not_reported")
    return {
        "plane_tilt_deg": angle,
        "plane_inlier_percent": plane_inlier_percent,
        "warnings": warnings,
    }


def _build_warnings(result: ProcessingResult, calibration_diagnostics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if result.status != "ok":
        warnings.append("processing_failed")
    if result.input_stats is None or result.input_stats.point_count < LOW_POINT_COUNT:
        warnings.append("low_point_count")
    if result.plane_model is None:
        warnings.append("missing_plane")
    if result.plane_mode == "auto" or not result.calibration_id:
        warnings.append("automatic_plane_estimation_only")
    total_objects = len(result.objects) + len(result.rejected_objects)
    if total_objects == 0:
        warnings.append("no_objects_detected")
    if total_objects > 0 and len(result.rejected_objects) / total_objects > HIGH_REJECTION_RATE:
        warnings.append("high_rejection_rate")
    total_ms = result.profiling.total_ms if result.profiling else float(result.timing_ms.total)
    if total_ms > ABNORMAL_TOTAL_MS:
        warnings.append("abnormal_processing_time")
    if calibration_diagnostics.get("warnings"):
        warnings.append("suspicious_calibration")
    return sorted(set(warnings))


def _slowest_stage(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    stages = ((result_payload.get("profiling") or {}).get("stages") or [])
    if not stages:
        return None
    stage = max(stages, key=lambda item: float(item.get("duration_ms") or 0.0))
    return {
        "name": stage.get("name"),
        "category": stage.get("category"),
        "duration_ms": stage.get("duration_ms"),
    }


def _plane_inlier_percent(result_payload: dict[str, Any]) -> float | None:
    plane_filtering = result_payload.get("plane_filtering")
    if isinstance(plane_filtering, dict):
        input_points = float(plane_filtering.get("input_points") or 0.0)
        foreground = float(plane_filtering.get("candidate_foreground_points") or 0.0)
        rejected = float(plane_filtering.get("rejected_points") or 0.0)
        if input_points > 0:
            return round(max(0.0, input_points - foreground - rejected) / input_points * 100.0, 2)
    for stage in ((result_payload.get("profiling") or {}).get("stages") or []):
        metadata = stage.get("metadata") or {}
        input_points = float(stage.get("input_points") or 0.0)
        plane_points = float(metadata.get("plane_points") or 0.0)
        if input_points > 0 and plane_points > 0:
            return round(plane_points / input_points * 100.0, 2)
    return None


def _plane_angle_deg(plane_model: tuple[float, float, float, float] | list[float] | None) -> float | None:
    if not plane_model:
        return None
    import math

    a, b, c, _ = [float(value) for value in plane_model]
    norm = math.sqrt(a * a + b * b + c * c)
    if norm == 0:
        return None
    cos_value = min(1.0, max(-1.0, abs(c) / norm))
    return round(math.degrees(math.acos(cos_value)), 3)


def _fit_score(item: dict[str, Any]) -> float | None:
    score = item.get("sphericity_score")
    if score is not None:
        return float(score)
    fit_rmse = item.get("fit_rmse_mm")
    diameter = item.get("diameter_mm") or item.get("diameter_estimate_mm")
    if fit_rmse is not None and diameter:
        return round(max(0.0, 1.0 - float(fit_rmse) / max(float(diameter), 0.001)), 4)
    return None


def _debug_images(result_payload: dict[str, Any], output_dir: Path | None) -> dict[str, bool]:
    files = result_payload.get("files") or {}
    return {
        key: _artifact_present(value, output_dir)
        for key, value in files.items()
        if key.startswith("debug_") or key in {"input_preview", "overlay"}
    }


def _artifact_present(filename: str | None, output_dir: Path | None) -> bool:
    if not filename:
        return False
    return (output_dir / filename).is_file() if output_dir is not None else True


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _infer_engine(result_payload: dict[str, Any]) -> str:
    stage = str(result_payload.get("algorithm_stage") or "")
    return "native" if stage == "classification" else "legacy"
