from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from PIL import Image

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.contracts.artifacts import normalize_processing_artifacts
from vision_3d_acquisition.pipelines.processing_units import (
    PROCESSING_UNIT_REGISTRY_VERSION,
    build_processing_unit_trace,
    build_recipe_snapshot,
    processing_unit_contract_fingerprint,
    recipe_parameters_by_unit,
)
from vision_3d_acquisition.pipelines.recipes import RecipeService
from vision_3d_acquisition.pipelines.registry import list_processing_unit_definitions
from vision_3d_acquisition.processing.status_index import process_entries_for_take


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _comparison_root(settings: ApiSettings, comparison_id: str) -> Path:
    root = settings.data_dir / "comparisons" / comparison_id
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    return root


def _safe_segment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value) or "artifact"


def comparison_file_url(comparison_id: str, filename: str) -> str:
    return f"/api/comparisons/{comparison_id}/files/{filename}"


def take_file_url(take_id: str, filename: str) -> str:
    return f"/api/takes/{take_id}/files/{filename}"


def resolve_comparison_file(settings: ApiSettings, comparison_id: str, filename: str) -> Path | None:
    root = (settings.data_dir / "comparisons" / comparison_id).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _pipeline_matches(payload: Mapping[str, Any] | None, pipeline_id: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    pipeline = payload.get("processing_pipeline")
    if isinstance(pipeline, Mapping):
        payload_pipeline_id = str(pipeline.get("id") or pipeline.get("name") or "")
        if payload_pipeline_id:
            return payload_pipeline_id == pipeline_id
    return pipeline_id == "mining_steel_ball_classification_25d"


def _resolve_run_payload(
    settings: ApiSettings,
    *,
    take_id: str,
    pipeline_id: str,
    run_id: str,
) -> tuple[dict[str, Any], Path] | None:
    processed_root = settings.processed_dir / take_id
    default_result = processed_root / "result.json"
    candidates: list[tuple[str, Path, dict[str, Any]]] = []

    if default_result.is_file():
        payload = _read_json(default_result)
        if payload is not None and _pipeline_matches(payload, pipeline_id):
            candidates.append(("processed", default_result, payload))

    entries = sorted(process_entries_for_take(settings.data_dir, take_id), key=lambda item: str(item.get("created_at") or ""), reverse=True)
    for entry in entries:
        entry_run_id = str(entry.get("run_id") or "")
        entry_pipeline_id = str(entry.get("pipeline_id") or pipeline_id)
        if entry_pipeline_id not in {pipeline_id, ""}:
            continue
        run_path = entry.get("path")
        if not run_path:
            continue
        result_path = Path(str(run_path)) / "result.json"
        if not result_path.is_file():
            continue
        payload = _read_json(result_path)
        if payload is None or not _pipeline_matches(payload, pipeline_id):
            continue
        candidates.append((entry_run_id or result_path.parent.name, result_path, payload))

    if not candidates:
        return None

    selected: tuple[str, Path, dict[str, Any]] | None
    if run_id == "latest":
        selected = candidates[0]
    else:
        selected = next((item for item in candidates if item[0] == run_id), None)
        if selected is None:
            selected = next((item for item in candidates if item[1].parent.name == run_id), None)
    if selected is None:
        return None

    _, result_path, payload = selected
    return payload, result_path.parent


def _ensure_run_contract_metadata(
    pipeline_id: str,
    result_payload: dict[str, Any],
    *,
    output_dir: Path,
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _json_copy(result_payload)
    payload["artifacts"] = normalize_processing_artifacts(payload, output_dir=output_dir)

    processing_pipeline = payload.get("processing_pipeline")
    if not isinstance(processing_pipeline, dict):
        processing_pipeline = {}
        payload["processing_pipeline"] = processing_pipeline
    if not isinstance(processing_pipeline.get("processing_units"), list):
        processing_pipeline["processing_units"] = units
    processing_pipeline.setdefault("id", pipeline_id)
    processing_pipeline.setdefault("version", "1.0")

    stage_params = payload.get("stage_params") if isinstance(payload.get("stage_params"), Mapping) else {}
    if not isinstance(payload.get("recipe_snapshot"), dict):
        payload["recipe_snapshot"] = build_recipe_snapshot(
            pipeline_id=pipeline_id,
            pipeline_version=str(processing_pipeline.get("version") or "1.0"),
            registry_version=PROCESSING_UNIT_REGISTRY_VERSION,
            contract_fingerprint=processing_unit_contract_fingerprint(units),
            units=units,
            stage_params=stage_params,
            recipe_version_id=payload.get("recipe_version_id") if isinstance(payload.get("recipe_version_id"), str) else None,
            enabled_units=[str(unit.get("id")) for unit in units if unit.get("id")],
            provenance={"source": "comparison_backfill"},
            artifact_policy=payload.get("artifact_policy") if isinstance(payload.get("artifact_policy"), dict) else None,
            calibration_snapshot_reference=(
                payload.get("calibration_snapshot_reference")
                if isinstance(payload.get("calibration_snapshot_reference"), dict)
                else None
            ),
        )
    payload["processing_unit_trace"] = build_processing_unit_trace(
        pipeline_id=pipeline_id,
        units=units,
        artifacts=list(payload.get("artifacts") or []),
        stage_params=stage_params,
        runtime_trace=payload.get("processing_unit_trace") if isinstance(payload.get("processing_unit_trace"), Mapping) else None,
    )
    return payload


def _source_parameters_by_unit(
    source: Mapping[str, Any],
    units: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    existing = source.get("parameters_by_unit")
    if isinstance(existing, Mapping):
        return {
            str(unit_id): dict(values)
            for unit_id, values in existing.items()
            if isinstance(values, Mapping)
        }
    stage_params = source.get("stage_params")
    if isinstance(stage_params, Mapping):
        return recipe_parameters_by_unit(units, stage_params)
    return {}


def _artifacts_by_id(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    return {
        str(artifact.get("artifact_id")): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("artifact_id")
    }


def _artifact_absolute_path(source: Mapping[str, Any], artifact: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(artifact, Mapping):
        return None
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    root = source.get("artifact_root")
    if not isinstance(root, Path):
        return None
    resolved = (root / path_value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved if resolved.is_file() else None


def _load_image_grayscale(path: Path) -> tuple[np.ndarray | None, tuple[int, int] | None, str | None]:
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            array = np.asarray(grayscale, dtype=np.uint8)
            width, height = grayscale.size
    except Exception as exc:
        return None, None, f"unreadable:{exc.__class__.__name__}"
    return array, (width, height), None


def _write_mask_image(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _write_overlay_image(path: Path, left_mask: np.ndarray, right_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = left_mask & right_mask
    removed = left_mask & (~right_mask)
    added = right_mask & (~left_mask)
    overlay = np.zeros((left_mask.shape[0], left_mask.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = kept.astype(np.uint8) * 255
    overlay[..., 0] = removed.astype(np.uint8) * 255
    overlay[..., 2] = added.astype(np.uint8) * 255
    Image.fromarray(overlay, mode="RGB").save(path)


def _binary_mask_diff(
    settings: ApiSettings,
    *,
    comparison_id: str,
    unit_id: str,
    artifact_id: str,
    left_source: Mapping[str, Any],
    right_source: Mapping[str, Any],
    left_artifact: Mapping[str, Any] | None,
    right_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left_path = _artifact_absolute_path(left_source, left_artifact)
    right_path = _artifact_absolute_path(right_source, right_artifact)
    result: dict[str, Any] = {
        "same_dimensions": None,
        "diff_available": False,
        "diff_type": "binary_mask",
        "diff_reason": None,
        "left_dimensions": None,
        "right_dimensions": None,
        "pixel_diff": None,
        "diff_artifacts": {},
    }
    if left_path is None or right_path is None:
        result["diff_reason"] = "missing_file"
        return result
    left_image, left_dims, left_error = _load_image_grayscale(left_path)
    right_image, right_dims, right_error = _load_image_grayscale(right_path)
    result["left_dimensions"] = {"width": left_dims[0], "height": left_dims[1]} if left_dims else None
    result["right_dimensions"] = {"width": right_dims[0], "height": right_dims[1]} if right_dims else None
    if left_error or right_error or left_image is None or right_image is None:
        result["diff_reason"] = left_error or right_error or "unreadable"
        return result
    if left_image.shape != right_image.shape:
        result["same_dimensions"] = False
        result["diff_reason"] = "dimension_mismatch"
        return result
    result["same_dimensions"] = True
    left_mask = left_image > 0
    right_mask = right_image > 0
    changed = left_mask ^ right_mask
    added = right_mask & (~left_mask)
    removed = left_mask & (~right_mask)
    intersection = left_mask & right_mask
    union = left_mask | right_mask
    total_pixels = int(left_mask.size)
    changed_pixels = int(np.count_nonzero(changed))
    added_pixels = int(np.count_nonzero(added))
    removed_pixels = int(np.count_nonzero(removed))
    intersection_pixels = int(np.count_nonzero(intersection))
    union_pixels = int(np.count_nonzero(union))
    left_active_pixels = int(np.count_nonzero(left_mask))
    right_active_pixels = int(np.count_nonzero(right_mask))
    unit_dir = _comparison_root(settings, comparison_id) / "artifacts" / _safe_segment(unit_id)
    base_name = _safe_segment(artifact_id)
    changed_path = unit_dir / f"{base_name}_changed.png"
    added_path = unit_dir / f"{base_name}_added.png"
    removed_path = unit_dir / f"{base_name}_removed.png"
    overlay_path = unit_dir / f"{base_name}_diff_overlay.png"
    _write_mask_image(changed_path, changed)
    _write_mask_image(added_path, added)
    _write_mask_image(removed_path, removed)
    _write_overlay_image(overlay_path, left_mask, right_mask)
    result["diff_available"] = True
    result["pixel_diff"] = {
        "width": int(left_mask.shape[1]),
        "height": int(left_mask.shape[0]),
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_percent": float((changed_pixels / total_pixels) * 100.0) if total_pixels else 0.0,
        "left_active_pixels": left_active_pixels,
        "right_active_pixels": right_active_pixels,
        "added_pixels": added_pixels,
        "removed_pixels": removed_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "iou": float(intersection_pixels / union_pixels) if union_pixels else 1.0,
        "added_percent_of_union": float((added_pixels / union_pixels) * 100.0) if union_pixels else 0.0,
        "removed_percent_of_union": float((removed_pixels / union_pixels) * 100.0) if union_pixels else 0.0,
    }
    result["diff_artifacts"] = {
        "changed_mask": {
            "path": f"artifacts/{_safe_segment(unit_id)}/{base_name}_changed.png",
            "url": comparison_file_url(comparison_id, f"artifacts/{_safe_segment(unit_id)}/{base_name}_changed.png"),
        },
        "added_mask": {
            "path": f"artifacts/{_safe_segment(unit_id)}/{base_name}_added.png",
            "url": comparison_file_url(comparison_id, f"artifacts/{_safe_segment(unit_id)}/{base_name}_added.png"),
        },
        "removed_mask": {
            "path": f"artifacts/{_safe_segment(unit_id)}/{base_name}_removed.png",
            "url": comparison_file_url(comparison_id, f"artifacts/{_safe_segment(unit_id)}/{base_name}_removed.png"),
        },
        "overlay": {
            "path": f"artifacts/{_safe_segment(unit_id)}/{base_name}_diff_overlay.png",
            "url": comparison_file_url(comparison_id, f"artifacts/{_safe_segment(unit_id)}/{base_name}_diff_overlay.png"),
        },
    }
    return result


def _trace_for_unit(source: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    trace = source.get("processing_unit_trace")
    if not isinstance(trace, Mapping):
        return {}
    unit_results = trace.get("unit_results")
    if not isinstance(unit_results, Mapping):
        unit_results = trace.get("units")
    if not isinstance(unit_results, Mapping):
        return {}
    entry = unit_results.get(unit_id)
    return dict(entry) if isinstance(entry, Mapping) else {}


def _classification_summary(result_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result_payload, Mapping):
        return {}
    classification = result_payload.get("classification")
    summary = result_payload.get("summary")
    return {
        "label": (
            (classification.get("label") if isinstance(classification, Mapping) else None)
            or (classification.get("class_label") if isinstance(classification, Mapping) else None)
            or (summary.get("class_label") if isinstance(summary, Mapping) else None)
            or (summary.get("processed_class_label") if isinstance(summary, Mapping) else None)
        ),
        "superclass": (
            (classification.get("superclass") if isinstance(classification, Mapping) else None)
            or (summary.get("processed_superclass") if isinstance(summary, Mapping) else None)
        ),
        "confidence": (
            (classification.get("confidence") if isinstance(classification, Mapping) else None)
            or (summary.get("confidence") if isinstance(summary, Mapping) else None)
        ),
    }


def _warning_count(result_payload: Mapping[str, Any] | None) -> int:
    if not isinstance(result_payload, Mapping):
        return 0
    pipeline_execution = result_payload.get("pipeline_execution")
    if isinstance(pipeline_execution, Mapping) and isinstance(pipeline_execution.get("stages"), list):
        return sum(
            len(stage.get("warnings") or [])
            for stage in pipeline_execution.get("stages") or []
            if isinstance(stage, Mapping)
        )
    warnings = result_payload.get("warnings")
    return len(warnings) if isinstance(warnings, list) else 0


def _top_level_metrics(source: Mapping[str, Any]) -> dict[str, Any]:
    result_payload = source.get("result_payload")
    if not isinstance(result_payload, Mapping):
        return {}
    objects = result_payload.get("objects")
    classification = _classification_summary(result_payload)
    return {
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "processing_status": str(result_payload.get("status") or "unknown"),
        "warning_count": _warning_count(result_payload),
        **classification,
    }


def _unit_metric_values(unit: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    result_payload = source.get("result_payload")
    trace = _trace_for_unit(source, str(unit.get("id") or ""))
    values: dict[str, Any] = {}
    if trace:
        values["status"] = trace.get("status")
        values["duration_ms"] = trace.get("duration_ms")
        values["warning_count"] = len(trace.get("warnings") or [])
        values["error_count"] = len(trace.get("errors") or [])
        diagnostics = trace.get("diagnostics")
        diagnostic_artifact_ids = diagnostics.get("artifact_ids") if isinstance(diagnostics, Mapping) else []
        values["diagnostic_artifact_count"] = len(diagnostic_artifact_ids or [])
        values["output_artifact_count"] = len(trace.get("output_artifacts") or [])
        metrics = trace.get("metrics")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                values[str(key)] = value
    if isinstance(result_payload, Mapping) and str(unit.get("kind") or "") == "stage":
        stage_id = str(unit.get("stage_id") or "")
        if stage_id in {"remove_belt_segment_objects", "geometry", "measurement"}:
            objects = result_payload.get("objects")
            values["object_count"] = len(objects) if isinstance(objects, list) else 0
        if stage_id == "classification":
            values.update(_classification_summary(result_payload))
        if stage_id == "detect_belt_plane":
            summary = result_payload.get("summary")
            if isinstance(summary, Mapping):
                if summary.get("plane_found") is not None:
                    values["plane_found"] = summary.get("plane_found")
    return values


def _diff_named_values(
    left_values: Mapping[str, Any],
    right_values: Mapping[str, Any],
    *,
    labels: Mapping[str, str] | None = None,
    include_unchanged: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left_values) | set(right_values)):
        left = left_values.get(key)
        right = right_values.get(key)
        status = "unchanged"
        if key not in left_values:
            status = "added"
        elif key not in right_values:
            status = "removed"
        elif _stable(left) != _stable(right):
            status = "changed"
        if not include_unchanged and status == "unchanged":
            continue
        rows.append(
            {
                "id": key,
                "label": (labels or {}).get(key, key),
                "left": left,
                "right": right,
                "status": status,
            }
        )
    return rows


def _coerce_source(
    settings: ApiSettings,
    *,
    pipeline_id: str,
    source: Mapping[str, Any],
    units: list[dict[str, Any]],
    recipe_service: RecipeService,
) -> dict[str, Any]:
    source_type = str(source.get("type") or "current")
    if source_type == "recipe":
        recipe_id = str(source.get("recipe_id") or "")
        version = source.get("version")
        recipe = recipe_service.get_recipe(pipeline_id, recipe_id, version=int(version) if isinstance(version, int) else None)
        return {
            "type": "recipe",
            "label": str(source.get("label") or recipe.get("name") or recipe_id),
            "recipe_id": recipe.get("recipe_id"),
            "recipe_version": recipe.get("version"),
            "parameters_by_unit": _source_parameters_by_unit(recipe, units),
            "stage_params": dict(recipe.get("stage_params") or {}),
            "artifacts": [],
            "artifact_root": None,
            "warnings": [],
        }
    if source_type == "run":
        take_id = str(source.get("take_id") or "")
        run_id = str(source.get("run_id") or "latest")
        resolved = _resolve_run_payload(settings, take_id=take_id, pipeline_id=pipeline_id, run_id=run_id)
        if resolved is None:
            raise KeyError(f"Unknown run for take {take_id}: {run_id}")
        result_payload, output_dir = resolved
        enriched = _ensure_run_contract_metadata(pipeline_id, result_payload, output_dir=output_dir, units=units)
        recipe_snapshot = enriched.get("recipe_snapshot") if isinstance(enriched.get("recipe_snapshot"), Mapping) else {}
        return {
            "type": "run",
            "label": str(source.get("label") or f"{take_id}:{run_id}"),
            "take_id": take_id,
            "run_id": run_id,
            "recipe_id": recipe_snapshot.get("recipe_id"),
            "recipe_version": recipe_snapshot.get("version"),
            "parameters_by_unit": _source_parameters_by_unit(recipe_snapshot, units),
            "stage_params": dict(enriched.get("stage_params") or {}),
            "artifacts": list(enriched.get("artifacts") or []),
            "artifact_root": output_dir,
            "processing_unit_trace": dict(enriched.get("processing_unit_trace") or {}),
            "result_payload": enriched,
            "warnings": [],
        }
    stage_params = source.get("stage_params") if isinstance(source.get("stage_params"), Mapping) else {}
    return {
        "type": "current",
        "label": str(source.get("label") or "Current edits"),
        "parameters_by_unit": _source_parameters_by_unit(dict(source), units) if source.get("parameters_by_unit") else recipe_parameters_by_unit(units, stage_params),
        "stage_params": dict(stage_params or {}),
        "artifacts": [],
        "artifact_root": None,
        "warnings": [],
    }


def resolve_pipeline_run_source(
    settings: ApiSettings,
    *,
    pipeline_id: str,
    take_id: str,
    run_id: str = "latest",
) -> dict[str, Any] | None:
    units = list_processing_unit_definitions(pipeline_id)
    resolved = _resolve_run_payload(settings, take_id=take_id, pipeline_id=pipeline_id, run_id=run_id)
    if resolved is None:
        return None
    result_payload, output_dir = resolved
    enriched = _ensure_run_contract_metadata(pipeline_id, result_payload, output_dir=output_dir, units=units)
    recipe_snapshot = enriched.get("recipe_snapshot") if isinstance(enriched.get("recipe_snapshot"), Mapping) else {}
    return {
        "type": "run",
        "label": f"{take_id}:{run_id}",
        "take_id": take_id,
        "run_id": run_id,
        "recipe_id": recipe_snapshot.get("recipe_id"),
        "recipe_version": recipe_snapshot.get("version"),
        "parameters_by_unit": _source_parameters_by_unit(recipe_snapshot, units),
        "stage_params": dict(enriched.get("stage_params") or {}),
        "artifacts": list(enriched.get("artifacts") or []),
        "artifact_root": output_dir,
        "processing_unit_trace": dict(enriched.get("processing_unit_trace") or {}),
        "result_payload": enriched,
        "warnings": [],
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _comparison_index_path(settings: ApiSettings) -> Path:
    root = settings.data_dir / "comparisons"
    root.mkdir(parents=True, exist_ok=True)
    return root / "index.json"


def _load_comparison_index(settings: ApiSettings) -> list[dict[str, Any]]:
    path = _comparison_index_path(settings)
    if not path.is_file():
        return []
    payload = _read_json(path)
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        return []
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def _write_comparison_index(settings: ApiSettings, entries: list[dict[str, Any]]) -> None:
    path = _comparison_index_path(settings)
    path.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")


def _comparison_index_entry(result: Mapping[str, Any], comparison_json_path: str) -> dict[str, Any]:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    return {
        "comparison_id": result.get("comparison_id"),
        "created_at": _now_iso(),
        "pipeline_id": result.get("pipeline_id"),
        "left": _json_copy(result.get("left")),
        "right": _json_copy(result.get("right")),
        "summary": {
            "parameter_changes": summary.get("parameter_changes"),
            "artifact_changes": summary.get("artifact_changes"),
            "metric_changes": summary.get("metric_changes"),
            "classification_changed": summary.get("classification_changed"),
            "object_count_changed": summary.get("object_count_changed"),
            "warnings_changed": summary.get("warnings_changed"),
            "what_changed_most": summary.get("what_changed_most"),
        },
        "comparison_json_path": comparison_json_path,
    }


def _append_comparison_index(settings: ApiSettings, entry: dict[str, Any]) -> None:
    entries = _load_comparison_index(settings)
    comparison_id = str(entry.get("comparison_id") or "")
    entries = [item for item in entries if str(item.get("comparison_id") or "") != comparison_id]
    entries.insert(0, entry)
    _write_comparison_index(settings, entries[:200])


def list_pipeline_comparisons(
    settings: ApiSettings,
    pipeline_id: str,
    *,
    take_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    entries = [
        item for item in _load_comparison_index(settings)
        if str(item.get("pipeline_id") or "") == pipeline_id
    ]
    if take_id:
        filtered: list[dict[str, Any]] = []
        for item in entries:
            left = item.get("left") if isinstance(item.get("left"), Mapping) else {}
            right = item.get("right") if isinstance(item.get("right"), Mapping) else {}
            if str(left.get("take_id") or "") == take_id or str(right.get("take_id") or "") == take_id:
                filtered.append(item)
        entries = filtered
    return entries[: max(1, min(limit, 100))]


def get_pipeline_comparison(settings: ApiSettings, pipeline_id: str, comparison_id: str) -> dict[str, Any]:
    path = settings.data_dir / "comparisons" / comparison_id / "comparison.json"
    if not path.is_file():
        raise KeyError(f"Unknown comparison: {comparison_id}")
    payload = _read_json(path)
    if payload is None:
        raise KeyError(f"Unreadable comparison: {comparison_id}")
    if str(payload.get("pipeline_id") or "") != pipeline_id:
        raise KeyError(f"Comparison {comparison_id} does not belong to pipeline {pipeline_id}")
    return payload


def _build_what_changed_most(
    *,
    comparison_units: Mapping[str, Any],
    largest_changed_artifact: dict[str, Any] | None,
    left_summary: Mapping[str, Any],
    right_summary: Mapping[str, Any],
    classification_changed: bool,
    warnings_changed: bool,
) -> dict[str, Any]:
    object_count_changed = _stable(left_summary.get("object_count")) != _stable(right_summary.get("object_count"))
    top_unit_id: str | None = None
    top_unit_label: str | None = None
    top_mask_changed_percent = 0.0
    top_unit_param_changes: list[dict[str, Any]] = []

    if largest_changed_artifact and largest_changed_artifact.get("unit_id"):
        top_unit_id = str(largest_changed_artifact.get("unit_id"))
        top_unit_label = str(largest_changed_artifact.get("unit_label") or top_unit_id)
        top_mask_changed_percent = float(largest_changed_artifact.get("changed_percent") or 0.0)
        unit = comparison_units.get(top_unit_id)
        if isinstance(unit, Mapping):
            top_unit_param_changes = [
                dict(row)
                for row in (unit.get("parameter_diff") or [])
                if isinstance(row, Mapping) and row.get("status") != "unchanged"
            ]
    else:
        best_score = -1.0
        for unit_id, unit in comparison_units.items():
            if not isinstance(unit, Mapping):
                continue
            unit_max = 0.0
            for art in unit.get("artifact_diff") or []:
                if not isinstance(art, Mapping):
                    continue
                pixel_diff = art.get("pixel_diff") if isinstance(art.get("pixel_diff"), Mapping) else None
                if pixel_diff:
                    unit_max = max(unit_max, float(pixel_diff.get("changed_percent") or 0.0))
            param_changes = [
                dict(row)
                for row in (unit.get("parameter_diff") or [])
                if isinstance(row, Mapping) and row.get("status") != "unchanged"
            ]
            score = unit_max if unit_max > 0 else float(len(param_changes))
            if score > best_score:
                best_score = score
                top_unit_id = unit_id
                top_unit_label = str(unit.get("label") or unit_id)
                top_mask_changed_percent = unit_max
                top_unit_param_changes = param_changes

    return {
        "top_changed_unit": {
            "unit_id": top_unit_id,
            "label": top_unit_label,
            "mask_changed_percent": top_mask_changed_percent if top_mask_changed_percent > 0 else None,
            "parameter_changes": [
                {
                    "parameter_id": row.get("parameter_id"),
                    "label": row.get("label"),
                    "left": row.get("left"),
                    "right": row.get("right"),
                }
                for row in top_unit_param_changes[:8]
            ],
        }
        if top_unit_id
        else None,
        "top_changed_artifact": largest_changed_artifact,
        "classification_changed": classification_changed,
        "object_count_changed": object_count_changed,
        "warnings_changed": warnings_changed,
    }


def compare_pipeline_sources(
    settings: ApiSettings,
    *,
    pipeline_id: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_id = f"cmp_{uuid4().hex[:12]}"
    units = list_processing_unit_definitions(pipeline_id)
    recipe_service = RecipeService(settings)
    left_source = _coerce_source(settings, pipeline_id=pipeline_id, source=left, units=units, recipe_service=recipe_service)
    right_source = _coerce_source(settings, pipeline_id=pipeline_id, source=right, units=units, recipe_service=recipe_service)
    unit_by_id = {str(unit.get("id") or ""): unit for unit in units if unit.get("id")}
    unit_order = {str(unit.get("id") or ""): int(unit.get("order") or 0) for unit in units if unit.get("id")}

    left_artifacts = _artifacts_by_id(left_source)
    right_artifacts = _artifacts_by_id(right_source)
    comparison_units: dict[str, Any] = {}
    parameter_changes = 0
    artifact_changes = 0
    metric_changes = 0
    changed_units: list[str] = []
    mask_artifacts_compared = 0
    mask_artifacts_changed = 0
    iou_values: list[float] = []
    largest_changed_artifact: dict[str, Any] | None = None
    comparison_warnings: list[str] = list(left_source.get("warnings") or []) + list(right_source.get("warnings") or [])

    for unit_id in sorted(unit_by_id, key=lambda item: (unit_order.get(item, 0), item)):
        unit = unit_by_id[unit_id]
        param_labels = {
            str(param.get("id") or ""): str(param.get("label") or param.get("id") or "")
            for param in unit.get("parameters") or []
            if isinstance(param, Mapping) and param.get("id")
        }
        left_params = dict((left_source.get("parameters_by_unit") or {}).get(unit_id) or {})
        right_params = dict((right_source.get("parameters_by_unit") or {}).get(unit_id) or {})
        left_trace = _trace_for_unit(left_source, unit_id)
        right_trace = _trace_for_unit(right_source, unit_id)
        if isinstance(left_trace.get("parameters_used"), Mapping):
            left_params = dict(left_trace.get("parameters_used") or {})
        if isinstance(right_trace.get("parameters_used"), Mapping):
            right_params = dict(right_trace.get("parameters_used") or {})
        parameter_diff = [
            {
                "parameter_id": row["id"],
                "label": row["label"],
                "left": row["left"],
                "right": row["right"],
                "status": row["status"],
            }
            for row in _diff_named_values(left_params, right_params, labels=param_labels)
        ]
        parameter_changes += sum(1 for row in parameter_diff if row["status"] != "unchanged")

        artifact_diff: list[dict[str, Any]] = []
        for artifact in unit.get("artifacts") or []:
            if not isinstance(artifact, Mapping) or not artifact.get("artifact_id"):
                continue
            artifact_id = str(artifact.get("artifact_id"))
            left_artifact = left_artifacts.get(artifact_id)
            right_artifact = right_artifacts.get(artifact_id)
            left_present = left_artifact is not None
            right_present = right_artifact is not None
            status = "unchanged" if left_present == right_present else "changed"
            artifact_row = {
                "artifact_id": artifact_id,
                "label": str(artifact.get("label") or artifact_id),
                "kind": str(artifact.get("kind") or artifact.get("renderer") or ""),
                "left_present": left_present,
                "right_present": right_present,
                "left_path": left_artifact.get("path") if left_artifact else None,
                "right_path": right_artifact.get("path") if right_artifact else None,
                "left_url": None,
                "right_url": None,
                "diffable": bool(artifact.get("diffable")),
                "same_dimensions": None,
                "left_dimensions": None,
                "right_dimensions": None,
                "diff_available": False,
                "diff_type": artifact.get("diff_type"),
                "diff_reason": None,
                "pixel_diff": None,
                "diff_artifacts": {},
                "status": status,
            }
            left_abs = _artifact_absolute_path(left_source, left_artifact)
            right_abs = _artifact_absolute_path(right_source, right_artifact)
            if left_present and left_abs is not None and isinstance(left_source.get("take_id"), str) and isinstance(artifact_row["left_path"], str):
                artifact_row["left_url"] = take_file_url(str(left_source["take_id"]), str(artifact_row["left_path"]))
            if right_present and right_abs is not None and isinstance(right_source.get("take_id"), str) and isinstance(artifact_row["right_path"], str):
                artifact_row["right_url"] = take_file_url(str(right_source["take_id"]), str(artifact_row["right_path"]))
            if bool(artifact.get("diffable")) and left_present and right_present and str(artifact.get("diff_type") or "") == "binary_mask":
                diff_payload = _binary_mask_diff(
                    settings,
                    comparison_id=comparison_id,
                    unit_id=unit_id,
                    artifact_id=artifact_id,
                    left_source=left_source,
                    right_source=right_source,
                    left_artifact=left_artifact,
                    right_artifact=right_artifact,
                )
                artifact_row.update(diff_payload)
                if artifact_row["diff_available"]:
                    mask_artifacts_compared += 1
                    pixel_diff = artifact_row.get("pixel_diff") if isinstance(artifact_row.get("pixel_diff"), Mapping) else {}
                    changed_percent = float(pixel_diff.get("changed_percent") or 0.0)
                    iou = float(pixel_diff.get("iou") or 0.0)
                    iou_values.append(iou)
                    if changed_percent > 0.0:
                        status = "changed"
                        artifact_row["status"] = "changed"
                        mask_artifacts_changed += 1
                        if largest_changed_artifact is None or changed_percent > float(largest_changed_artifact.get("changed_percent") or -1):
                            largest_changed_artifact = {
                                "artifact_id": artifact_id,
                                "label": artifact_row["label"],
                                "unit_id": unit_id,
                                "unit_label": str(unit.get("label") or unit_id),
                                "changed_percent": changed_percent,
                            }
                elif artifact_row.get("diff_reason"):
                    comparison_warnings.append(f"{unit_id}.{artifact_id}: {artifact_row['diff_reason']}")
                    if left_present and right_present:
                        artifact_row["status"] = "changed"
            if artifact_row["status"] == "changed":
                artifact_changes += 1
            artifact_diff.append(artifact_row)

        metric_labels = {
            "status": "Status",
            "warning_count": "Warning count",
            "error_count": "Error count",
            "duration_ms": "Duration (ms)",
            "diagnostic_artifact_count": "Diagnostic artifacts",
            "output_artifact_count": "Output artifacts",
            "object_count": "Object count",
            "plane_found": "Plane found",
            "label": "Classification label",
            "superclass": "Classification superclass",
            "confidence": "Classification confidence",
        }
        metric_diff = _diff_named_values(
            _unit_metric_values(unit, left_source),
            _unit_metric_values(unit, right_source),
            labels=metric_labels,
        )
        metric_changes += sum(1 for row in metric_diff if row["status"] != "unchanged")

        diagnostic_diff = _diff_named_values(
            {
                "trace_source": left_trace.get("trace_source"),
                "trace_precision": left_trace.get("trace_precision"),
                "status": left_trace.get("status"),
                "duration_ms": left_trace.get("duration_ms"),
                "warning_count": len(left_trace.get("warnings") or []),
                "error_count": len(left_trace.get("errors") or []),
                "diagnostic_artifact_count": len(((left_trace.get("diagnostics") or {}) if isinstance(left_trace.get("diagnostics"), Mapping) else {}).get("artifact_ids") or []),
            } if left_trace else {},
            {
                "trace_source": right_trace.get("trace_source"),
                "trace_precision": right_trace.get("trace_precision"),
                "status": right_trace.get("status"),
                "duration_ms": right_trace.get("duration_ms"),
                "warning_count": len(right_trace.get("warnings") or []),
                "error_count": len(right_trace.get("errors") or []),
                "diagnostic_artifact_count": len(((right_trace.get("diagnostics") or {}) if isinstance(right_trace.get("diagnostics"), Mapping) else {}).get("artifact_ids") or []),
            } if right_trace else {},
            labels={
                "trace_source": "Trace source",
                "trace_precision": "Trace precision",
                "status": "Status",
                "duration_ms": "Duration (ms)",
                "warning_count": "Warning count",
                "error_count": "Error count",
                "diagnostic_artifact_count": "Diagnostic artifacts",
            },
        )

        has_changes = (
            any(row["status"] != "unchanged" for row in parameter_diff)
            or any(row["status"] != "unchanged" for row in artifact_diff)
            or any(row["status"] != "unchanged" for row in metric_diff)
            or any(row["status"] != "unchanged" for row in diagnostic_diff)
        )
        if has_changes:
            changed_units.append(unit_id)

        comparison_units[unit_id] = {
            "label": str(unit.get("label") or unit_id),
            "stage_id": str(unit.get("stage_id") or ""),
            "kind": str(unit.get("kind") or ""),
            "order": unit_order.get(unit_id, 0),
            "has_changes": has_changes,
            "parameter_diff": parameter_diff,
            "artifact_diff": artifact_diff,
            "metric_diff": metric_diff,
            "diagnostic_diff": diagnostic_diff,
        }

    left_summary = _top_level_metrics(left_source)
    right_summary = _top_level_metrics(right_source)
    classification_changed = any(
        _stable(left_summary.get(key)) != _stable(right_summary.get(key))
        for key in ("label", "superclass", "confidence")
    )
    warnings_changed = left_summary.get("warning_count") != right_summary.get("warning_count")
    object_count_changed = _stable(left_summary.get("object_count")) != _stable(right_summary.get("object_count"))
    what_changed_most = _build_what_changed_most(
        comparison_units=comparison_units,
        largest_changed_artifact=largest_changed_artifact,
        left_summary=left_summary,
        right_summary=right_summary,
        classification_changed=classification_changed,
        warnings_changed=warnings_changed,
    )

    result = {
        "comparison_id": comparison_id,
        "pipeline_id": pipeline_id,
        "contract_fingerprint": processing_unit_contract_fingerprint(units),
        "left": {
            "type": left_source.get("type"),
            "label": left_source.get("label"),
            "take_id": left_source.get("take_id"),
            "run_id": left_source.get("run_id"),
            "recipe_id": left_source.get("recipe_id"),
            "recipe_version": left_source.get("recipe_version"),
        },
        "right": {
            "type": right_source.get("type"),
            "label": right_source.get("label"),
            "take_id": right_source.get("take_id"),
            "run_id": right_source.get("run_id"),
            "recipe_id": right_source.get("recipe_id"),
            "recipe_version": right_source.get("recipe_version"),
        },
        "summary": {
            "parameter_changes": parameter_changes,
            "artifact_changes": artifact_changes,
            "metric_changes": metric_changes,
            "classification_changed": classification_changed,
            "object_count_changed": object_count_changed,
            "warnings_changed": warnings_changed,
            "left_object_count": left_summary.get("object_count"),
            "right_object_count": right_summary.get("object_count"),
            "left_status": left_summary.get("processing_status"),
            "right_status": right_summary.get("processing_status"),
            "key_affected_units": changed_units[:8],
            "mask_artifacts_compared": mask_artifacts_compared,
            "mask_artifacts_changed": mask_artifacts_changed,
            "largest_changed_artifact": largest_changed_artifact.get("artifact_id") if largest_changed_artifact else None,
            "largest_changed_unit": largest_changed_artifact.get("unit_id") if largest_changed_artifact else None,
            "average_iou": float(sum(iou_values) / len(iou_values)) if iou_values else None,
            "what_changed_most": what_changed_most,
        },
        "units": comparison_units,
        "warnings": comparison_warnings,
    }
    comparison_root = _comparison_root(settings, comparison_id)
    comparison_json_path = f"comparisons/{comparison_id}/comparison.json"
    (comparison_root / "comparison.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    _append_comparison_index(settings, _comparison_index_entry(result, comparison_json_path))
    return result
