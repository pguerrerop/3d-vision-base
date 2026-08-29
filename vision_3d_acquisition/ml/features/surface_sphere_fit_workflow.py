from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vision_3d_acquisition.api.feature_catalog import (
    feature_definition_for_key,
    feature_job_backfill_keys,
    feature_job_reprocess_keys,
)
from vision_3d_acquisition.vision_core.geometry.surface_sphere_fit import (
    FEATURE_ALGORITHM_VERSION,
    build_visible_object_points_xyz_mm,
    mask_from_contour_px,
    surface_sphere_fit_rmse_mm,
)
from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, load_heightmap_npz

SUPPORTED_FEATURE_KEY = "surface_sphere_fit_rmse_mm"
BACKFILL_SIDECAR_NAME = "feature_backfill_surface_sphere_fit_rmse_mm.json"
PIPELINE_ID_25D = "mining_steel_ball_classification_25d"


def backfill_sidecar_name(feature_key: str) -> str:
    return f"feature_backfill_{feature_key}.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:
        return None
    return num


def _extract_numeric_path(raw_object: dict[str, Any], path: str) -> float | None:
    current: Any = raw_object
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _to_float(current)


def _feature_locations(feature_key: str = SUPPORTED_FEATURE_KEY) -> tuple[str, ...]:
    definition = feature_definition_for_key(feature_key)
    return tuple(definition.extraction_paths)


def _load_contour_sidecar(output_dir: Path) -> dict[str, list[list[float]]]:
    payload = _read_json(output_dir / "contour.json")
    out: dict[str, list[list[float]]] = {}
    for item in payload.get("objects") or []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        contour = item.get("contour_px")
        if object_id and isinstance(contour, list) and len(contour) >= 3:
            out[object_id] = contour
    return out


def _resolve_object_contour_px(raw_object: dict[str, Any], contour_sidecar: dict[str, list[list[float]]]) -> list[list[float]] | None:
    contour = raw_object.get("contour_px")
    if isinstance(contour, list) and len(contour) >= 3:
        return contour
    object_id = str(raw_object.get("object_id") or "")
    sidecar_contour = contour_sidecar.get(object_id)
    if isinstance(sidecar_contour, list) and len(sidecar_contour) >= 3:
        return sidecar_contour
    return None


def _find_feature_in_object(raw_object: dict[str, Any], feature_key: str = SUPPORTED_FEATURE_KEY) -> tuple[float | None, str | None]:
    for path in _feature_locations(feature_key):
        parsed = _extract_numeric_path(raw_object, path) if "." in path else _to_float(raw_object.get(path))
        if parsed is not None:
            return parsed, f"result.json objects[] ({path})"
    return None, None


def _resolve_normalized_heightmap_path(output_dir: Path, result: dict[str, Any]) -> Path | None:
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    rel = files.get("normalized_heightmap") or "normalized_heightmap.npz"
    candidate = output_dir / str(rel)
    if candidate.is_file():
        return candidate
    fallback = output_dir / "normalized_heightmap.npz"
    return fallback if fallback.is_file() else None


def _load_normalized_frame(output_dir: Path, result: dict[str, Any]) -> tuple[HeightmapFrame | None, np.ndarray | None]:
    path = _resolve_normalized_heightmap_path(output_dir, result)
    if path is None:
        return None, None
    frame = load_heightmap_npz(path)
    return frame, np.asarray(frame.z_mm, dtype=np.float32)


def _build_provenance_entry(
    *,
    take_id: str,
    run_id: str,
    pipeline_id: str,
    object_id: str,
    physical_object_id: str | None,
    calibration_id: str | None,
    value: float | None,
    source: str,
    diagnostic_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "feature_key": SUPPORTED_FEATURE_KEY,
        "feature_algorithm_version": FEATURE_ALGORITHM_VERSION,
        "source": source,
        "pipeline_id": pipeline_id,
        "run_id": run_id,
        "take_id": take_id,
        "object_id": object_id,
        "physical_object_id": physical_object_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "calibration_id": calibration_id,
        "geometry_coordinate_space": "corrected_metric_mm",
        "input_artifacts": [
            "normalized_heightmap",
            "object_mask",
            "contour.json",
            "measurement_metadata",
        ],
        "value": value,
        "diagnostic_reason": diagnostic_reason,
    }


def inspect_take_feature(
    *,
    processed_dir: Path,
    take_id: str,
    feature_key: str = SUPPORTED_FEATURE_KEY,
) -> dict[str, Any]:
    supported = feature_job_reprocess_keys() | feature_job_backfill_keys()
    if feature_key not in supported:
        raise ValueError(f"unsupported feature for inspect workflow: {feature_key}")

    extraction_paths = _feature_locations(feature_key)
    output_dir = processed_dir / take_id
    result_path = output_dir / "result.json"
    result = _read_json(result_path)
    if not result:
        return {
            "take_id": take_id,
            "found": False,
            "reason": "no_processed_output",
            "result_path": str(result_path),
            "locations_checked": [],
        }

    processing_pipeline = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
    pipeline_id = str(processing_pipeline.get("id") or "")
    run_id = str(result.get("run_id") or "")
    calibration_id = str(result.get("calibration_id") or "") or None
    objects = [item for item in (result.get("objects") or []) if isinstance(item, dict)]
    feature_vector_path = output_dir / "feature_vector.json"
    diagnostics_path = output_dir / "measurement_diagnostics.json"
    backfill_path = output_dir / backfill_sidecar_name(feature_key)
    contour_sidecar = _load_contour_sidecar(output_dir)

    locations_checked = [f"result.json objects[] ({path})" for path in extraction_paths] + [
        f"feature_vector.json features.{feature_key}",
        "measurement_diagnostics.json",
        "contour.json",
        backfill_sidecar_name(feature_key),
    ]

    values: list[float] = []
    per_object: list[dict[str, Any]] = []
    missing_reasons: list[str] = []

    feature_vector = _read_json(feature_vector_path)
    fv_value = _to_float(((feature_vector.get("features") or {}) if isinstance(feature_vector.get("features"), dict) else {}).get(feature_key))
    if fv_value is not None:
        values.append(fv_value)

    backfill_payload = _read_json(backfill_path)
    backfill_entries = [item for item in (backfill_payload.get("entries") or []) if isinstance(item, dict)]
    backfill_by_object = {
        str(item.get("object_id") or ""): item
        for item in backfill_entries
        if str(item.get("object_id") or "")
    }
    backfill_values = [_to_float(item.get("value")) for item in backfill_entries]
    backfill_values = [value for value in backfill_values if value is not None]

    for raw in objects:
        object_id = str(raw.get("object_id") or "")
        found_value, found_location = _find_feature_in_object(raw, feature_key)
        backfill_entry = backfill_by_object.get(object_id)
        backfill_value = _to_float((backfill_entry or {}).get("value"))
        entry = {
            "object_id": object_id,
            "value": found_value if found_value is not None else backfill_value,
            "location": found_location,
            "backfill_value": backfill_value,
            "backfill_source": (backfill_entry or {}).get("source"),
            "missing_reason": None,
        }
        if found_value is None and backfill_value is not None:
            entry["location"] = backfill_sidecar_name(feature_key)
            values.append(backfill_value)
        elif found_value is not None:
            values.append(found_value)

        if entry["value"] is None:
            point_count = int(raw.get("point_count") or 0)
            contour = _resolve_object_contour_px(raw, contour_sidecar)
            if point_count > 0 and point_count < 8:
                entry["missing_reason"] = "insufficient_object_points"
            elif not contour:
                entry["missing_reason"] = "missing_object_contour"
            elif not (raw.get("surface_geometry") or {}) and not (raw.get("sphere_consistency") or {}):
                entry["missing_reason"] = "old_run_before_surface_geometry_emitted"
            else:
                sphere_group = raw.get("surface_geometry") if isinstance(raw.get("surface_geometry"), dict) else {}
                consistency_group = raw.get("sphere_consistency") if isinstance(raw.get("sphere_consistency"), dict) else {}
                if feature_key == SUPPORTED_FEATURE_KEY and sphere_group.get("sphere_fit_rmse_mm") is None:
                    entry["missing_reason"] = "sphere_fit_invalid_or_insufficient_points"
                elif feature_key.startswith("surface_sphere_") and not consistency_group:
                    entry["missing_reason"] = "old_run_before_sphere_consistency_emitted"
                else:
                    entry["missing_reason"] = "feature_not_emitted_by_pipeline"
            missing_reasons.append(entry["missing_reason"])
        per_object.append(entry)

    summary = {
        "take_id": take_id,
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "result_path": str(result_path),
        "object_count": len(objects),
        "objects_with_feature_value": sum(1 for item in per_object if item.get("value") is not None),
        "objects_missing_feature": sum(1 for item in per_object if item.get("value") is None),
        "backfill_object_count": len(backfill_values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": float(sum(values) / len(values)) if values else None,
        "feature_vector_value": fv_value,
        "feature_vector_path": str(feature_vector_path) if feature_vector_path.is_file() else None,
        "measurement_diagnostics_path": str(diagnostics_path) if diagnostics_path.is_file() else None,
        "backfill_sidecar_path": str(backfill_path) if backfill_path.is_file() else None,
        "locations_checked": locations_checked,
        "per_object": per_object,
        "likely_reason": None,
    }
    if not values and not backfill_values:
        if not pipeline_id:
            summary["likely_reason"] = "no_processed_output"
        elif pipeline_id != PIPELINE_ID_25D:
            summary["likely_reason"] = "pipeline_not_25d_measurement"
        elif all(reason == "old_run_before_surface_geometry_emitted" for reason in missing_reasons if reason):
            summary["likely_reason"] = "old_run_before_feature_existed"
        elif any(reason == "insufficient_object_points" for reason in missing_reasons):
            summary["likely_reason"] = "insufficient_object_points"
        else:
            summary["likely_reason"] = "feature_not_emitted_by_pipeline"
    summary["found"] = bool(values or backfill_values)
    return summary


def backfill_take_feature(
    *,
    processed_dir: Path,
    take_id: str,
    incoming_dir: Path | None = None,
    feature_key: str = SUPPORTED_FEATURE_KEY,
    dry_run: bool = True,
) -> dict[str, Any]:
    if feature_key != SUPPORTED_FEATURE_KEY:
        raise ValueError(f"unsupported feature for backfill workflow: {feature_key}")

    output_dir = processed_dir / take_id
    result = _read_json(output_dir / "result.json")
    if not result:
        raise ValueError(f"processed result not found for take {take_id}")

    frame, normalized = _load_normalized_frame(output_dir, result)
    if frame is None or normalized is None:
        raise ValueError(f"normalized heightmap not found for take {take_id}")

    processing_pipeline = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
    pipeline_id = str(processing_pipeline.get("id") or PIPELINE_ID_25D)
    run_id = str(result.get("run_id") or "")
    calibration_id = str(result.get("calibration_id") or "") or None
    physical_object_id = None
    if incoming_dir is not None:
        incoming_meta = _read_json(incoming_dir / take_id / "metadata.json")
        physical_object_id = str(incoming_meta.get("physical_object_id") or "") or None

    entries: list[dict[str, Any]] = []
    objects = [item for item in (result.get("objects") or []) if isinstance(item, dict)]
    contour_sidecar = _load_contour_sidecar(output_dir)
    for raw in objects:
        object_id = str(raw.get("object_id") or "")
        existing, _ = _find_feature_in_object(raw)
        if existing is not None:
            continue
        contour = _resolve_object_contour_px(raw, contour_sidecar)
        mask = mask_from_contour_px(contour, normalized.shape)
        if not np.any(mask):
            bbox = raw.get("bbox_px")
            if isinstance(bbox, list) and len(bbox) >= 4:
                x0, y0, w, h = [int(v) for v in bbox[:4]]
                mask[max(0, y0) : y0 + max(0, h), max(0, x0) : x0 + max(0, w)] = True
        points = build_visible_object_points_xyz_mm(normalized_heightmap_mm=normalized, frame=frame, mask=mask)
        rmse, fit = surface_sphere_fit_rmse_mm(points)
        entry = _build_provenance_entry(
            take_id=take_id,
            run_id=run_id,
            pipeline_id=pipeline_id,
            object_id=object_id,
            physical_object_id=physical_object_id,
            calibration_id=calibration_id,
            value=rmse,
            source="feature_backfill",
            diagnostic_reason=None if rmse is not None else str(fit.get("reason") or "compute_failed"),
        )
        entries.append(entry)

    payload = {
        "feature_key": SUPPORTED_FEATURE_KEY,
        "feature_algorithm_version": FEATURE_ALGORITHM_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "feature_backfill",
        "pipeline_id": pipeline_id,
        "run_id": run_id,
        "take_id": take_id,
        "entries": entries,
    }
    sidecar_path = output_dir / backfill_sidecar_name(feature_key)
    if not dry_run:
        sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "take_id": take_id,
        "dry_run": bool(dry_run),
        "sidecar_path": str(sidecar_path),
        "entries_written": len(entries),
        "objects_with_value": sum(1 for item in entries if item.get("value") is not None),
        "objects_missing_value": sum(1 for item in entries if item.get("value") is None),
        "payload": payload,
    }


def take_needs_feature_action(
    processed_dir: Path,
    take_id: str,
    feature_key: str,
    mode: str,
    *,
    only_missing: bool,
) -> bool:
    if not only_missing:
        return True
    summary = inspect_take_feature(processed_dir=processed_dir, take_id=take_id, feature_key=feature_key)
    per_object = summary.get("per_object") or []
    if not per_object:
        return True
    for item in per_object:
        location = str(item.get("location") or "")
        value = item.get("value")
        has_canonical = value is not None and location.startswith("result.json")
        if mode == "backfill":
            if not has_canonical:
                return True
        elif not has_canonical:
            return True
    return False


def load_feature_backfill_index(processed_dir: Path, take_id: str, feature_key: str) -> dict[str, dict[str, Any]]:
    if feature_key != SUPPORTED_FEATURE_KEY:
        return {}
    sidecar_name = backfill_sidecar_name(feature_key)
    payload = _read_json(processed_dir / take_id / sidecar_name)
    if str(payload.get("feature_key") or "") != feature_key:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("entries") or []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        value = _to_float(item.get("value"))
        if not object_id or value is None:
            continue
        out[object_id] = item
    return out


def compute_ml_set_feature(
    *,
    data_dir: Path,
    ml_set_id: str,
    feature_key: str,
    mode: str,
    dataset_id: str | None = None,
    pipeline_id: str = PIPELINE_ID_25D,
    dry_run: bool = True,
) -> dict[str, Any]:
    from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing  # noqa: WPS433
    from vision_3d_acquisition.api.settings import ApiSettings  # noqa: WPS433
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    if feature_key != SUPPORTED_FEATURE_KEY:
        raise ValueError(f"unsupported feature for compute workflow: {feature_key}")
    if mode not in {"reprocess", "backfill"}:
        raise ValueError("mode must be one of: reprocess, backfill")

    service = DatasetService(data_dir)
    ml_set = service.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
    resolved_dataset_id = str(ml_set.get("dataset_id") or "")
    memberships = service.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=ml_set_id)
    take_ids = sorted({str(row.get("take_id") or "") for row in memberships if row.get("include", True) and str(row.get("take_id") or "")})
    if not take_ids:
        raise ValueError("no included takes found for ML set")

    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()

    results: list[dict[str, Any]] = []
    for take_id in take_ids:
        if mode == "reprocess":
            if dry_run:
                results.append({"take_id": take_id, "status": "planned_reprocess", "pipeline_id": pipeline_id})
                continue
            dispatch = dispatch_take_processing(
                settings=settings,
                take_id=take_id,
                pipeline_id=pipeline_id,
                reprocess=True,
                source_id=None,
                recipe_version_id=None,
                acquisition_group_id=None,
                calibration_profile_id=None,
                stage_params=None,
            )
            inspect = inspect_take_feature(processed_dir=settings.processed_dir, take_id=take_id, feature_key=feature_key)
            results.append(
                {
                    "take_id": take_id,
                    "status": str(dispatch.get("status") or "ok"),
                    "source": "pipeline_reprocess",
                    "objects_with_feature_value": inspect.get("objects_with_feature_value"),
                }
            )
        else:
            item = backfill_take_feature(
                processed_dir=settings.processed_dir,
                take_id=take_id,
                incoming_dir=settings.incoming_dir,
                feature_key=feature_key,
                dry_run=dry_run,
            )
            item["status"] = "planned_backfill" if dry_run else "backfilled"
            results.append(item)

    return {
        "ml_set_id": ml_set_id,
        "dataset_id": resolved_dataset_id,
        "feature_key": feature_key,
        "mode": mode,
        "dry_run": bool(dry_run),
        "take_count": len(take_ids),
        "results": results,
    }
