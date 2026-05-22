from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.contracts.modalities import infer_modalities
from vision_3d_acquisition.processing.status_index import process_entries_for_take


FusionReadinessStatus = Literal[
    "waiting_for_rgb",
    "waiting_for_heightmap",
    "waiting_for_processing",
    "ready_for_fusion",
    "incomplete",
    "failed_input",
]

MatchingMethod = Literal[
    "centroid_projection",
    "bbox_overlap",
    "object_index_fallback",
    "unmatched_rgb",
    "unmatched_heightmap",
]


@dataclass
class FusionInputBundle:
    acquisition_group_id: str
    rgb_take_id: str | None = None
    heightmap_take_id: str | None = None
    rgb_run_id: str | None = None
    heightmap_run_id: str | None = None
    rgb_result_payload: dict[str, Any] | None = None
    heightmap_result_payload: dict[str, Any] | None = None
    rgb_artifacts: list[dict[str, Any]] = field(default_factory=list)
    heightmap_artifacts: list[dict[str, Any]] = field(default_factory=list)
    readiness_status: FusionReadinessStatus = "incomplete"
    missing_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    resolved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionObjectCandidate:
    fusion_object_id: str
    rgb_object_id: str | None
    heightmap_object_id: str | None
    matching_method: MatchingMethod
    confidence: float
    rgb_features: dict[str, Any] = field(default_factory=dict)
    heightmap_features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FusionPreviewResult:
    acquisition_group_id: str
    readiness_status: FusionReadinessStatus
    object_candidates: list[FusionObjectCandidate] = field(default_factory=list)
    merged_feature_summary: dict[str, Any] = field(default_factory=dict)
    recommended_next_action: str = "wait_for_inputs"
    warnings: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


def resolve_fusion_inputs(data_dir: Path, acquisition_group_id: str) -> FusionInputBundle:
    bundle = FusionInputBundle(acquisition_group_id=acquisition_group_id)
    try:
        group_takes = AcquisitionGroupService(data_dir).list_group_takes(acquisition_group_id)
    except Exception as exc:
        bundle.readiness_status = "failed_input"
        bundle.warnings.append(f"Failed to load acquisition group: {exc}")
        return bundle

    rgb_candidates: list[dict[str, Any]] = []
    heightmap_candidates: list[dict[str, Any]] = []
    for item in group_takes:
        take_id = str(item.get("take_id") or "")
        if not take_id:
            continue
        metadata = _read_json(data_dir / "incoming" / take_id / "metadata.json") or {}
        modalities = infer_modalities(metadata, data_dir / "incoming" / take_id)
        if "rgb" in modalities:
            rgb_candidates.append(_candidate_for_take(data_dir, take_id, "2d"))
        if "heightmap" in modalities:
            heightmap_candidates.append(_candidate_for_take(data_dir, take_id, "25d"))

    rgb_candidates = [item for item in rgb_candidates if item]
    heightmap_candidates = [item for item in heightmap_candidates if item]
    if len(rgb_candidates) > 1:
        bundle.warnings.append("Multiple RGB candidates found; selected latest successful run.")
    if len(heightmap_candidates) > 1:
        bundle.warnings.append("Multiple 25D candidates found; selected latest successful run.")

    rgb_selected = _select_best_candidate(rgb_candidates)
    heightmap_selected = _select_best_candidate(heightmap_candidates)
    bundle.debug = {
        "rgb_candidates": rgb_candidates,
        "heightmap_candidates": heightmap_candidates,
    }

    if rgb_selected:
        bundle.rgb_take_id = str(rgb_selected.get("take_id") or "")
        bundle.rgb_run_id = str(rgb_selected.get("run_id") or "") or None
        bundle.rgb_result_payload = rgb_selected.get("result_payload")
        bundle.rgb_artifacts = rgb_selected.get("artifacts") or []
    if heightmap_selected:
        bundle.heightmap_take_id = str(heightmap_selected.get("take_id") or "")
        bundle.heightmap_run_id = str(heightmap_selected.get("run_id") or "") or None
        bundle.heightmap_result_payload = heightmap_selected.get("result_payload")
        bundle.heightmap_artifacts = heightmap_selected.get("artifacts") or []

    has_rgb_take = bool(rgb_candidates)
    has_heightmap_take = bool(heightmap_candidates)
    has_rgb_run = bool(rgb_selected and rgb_selected.get("result_payload"))
    has_heightmap_run = bool(heightmap_selected and heightmap_selected.get("result_payload"))

    if not has_rgb_take and not has_heightmap_take:
        bundle.readiness_status = "incomplete"
        bundle.missing_inputs.extend(["rgb_take", "heightmap_take"])
    elif not has_rgb_take:
        bundle.readiness_status = "waiting_for_rgb"
        bundle.missing_inputs.append("rgb_take")
    elif not has_heightmap_take:
        bundle.readiness_status = "waiting_for_heightmap"
        bundle.missing_inputs.append("heightmap_take")
    elif not has_rgb_run or not has_heightmap_run:
        bundle.readiness_status = "waiting_for_processing"
        if not has_rgb_run:
            bundle.missing_inputs.append("rgb_processed_run")
        if not has_heightmap_run:
            bundle.missing_inputs.append("heightmap_processed_run")
    else:
        bundle.readiness_status = "ready_for_fusion"

    bundle.resolved_at = datetime.now(UTC).isoformat()
    return bundle


def build_fusion_preview(data_dir: Path, acquisition_group_id: str) -> FusionPreviewResult:
    bundle = resolve_fusion_inputs(data_dir, acquisition_group_id)
    result = FusionPreviewResult(
        acquisition_group_id=acquisition_group_id,
        readiness_status=bundle.readiness_status,
        warnings=list(bundle.warnings),
        inputs=asdict(bundle),
    )

    if bundle.readiness_status != "ready_for_fusion":
        result.recommended_next_action = _next_action_for_status(bundle.readiness_status)
        return result

    rgb_objects = _extract_objects(bundle.rgb_result_payload, "rgb", result.warnings)
    heightmap_objects = _extract_objects(bundle.heightmap_result_payload, "heightmap", result.warnings)
    max_count = max(len(rgb_objects), len(heightmap_objects))
    candidates: list[FusionObjectCandidate] = []
    for idx in range(max_count):
        rgb_obj = rgb_objects[idx] if idx < len(rgb_objects) else None
        hm_obj = heightmap_objects[idx] if idx < len(heightmap_objects) else None
        if rgb_obj and hm_obj:
            method: MatchingMethod = "object_index_fallback"
            confidence = 0.45
            if _has_centroid(rgb_obj) and _has_centroid(hm_obj):
                method = "centroid_projection"
                confidence = 0.60
            elif _has_bbox(rgb_obj) and _has_bbox(hm_obj):
                method = "bbox_overlap"
                confidence = 0.55
            candidates.append(
                FusionObjectCandidate(
                    fusion_object_id=f"fobj_{idx+1}",
                    rgb_object_id=rgb_obj.get("object_id"),
                    heightmap_object_id=hm_obj.get("object_id"),
                    matching_method=method,
                    confidence=confidence,
                    rgb_features=rgb_obj,
                    heightmap_features=hm_obj,
                    warnings=[],
                )
            )
        elif rgb_obj:
            candidates.append(
                FusionObjectCandidate(
                    fusion_object_id=f"fobj_{idx+1}",
                    rgb_object_id=rgb_obj.get("object_id"),
                    heightmap_object_id=None,
                    matching_method="unmatched_rgb",
                    confidence=0.0,
                    rgb_features=rgb_obj,
                    warnings=["No matching 25D object candidate."],
                )
            )
        elif hm_obj:
            candidates.append(
                FusionObjectCandidate(
                    fusion_object_id=f"fobj_{idx+1}",
                    rgb_object_id=None,
                    heightmap_object_id=hm_obj.get("object_id"),
                    matching_method="unmatched_heightmap",
                    confidence=0.0,
                    heightmap_features=hm_obj,
                    warnings=["No matching RGB object candidate."],
                )
            )

    result.object_candidates = candidates
    result.merged_feature_summary = {
        "rgb_object_count": len(rgb_objects),
        "heightmap_object_count": len(heightmap_objects),
        "fused_candidate_count": len(candidates),
        "matched_count": sum(1 for item in candidates if item.rgb_object_id and item.heightmap_object_id),
        "unmatched_rgb_count": sum(1 for item in candidates if item.matching_method == "unmatched_rgb"),
        "unmatched_heightmap_count": sum(1 for item in candidates if item.matching_method == "unmatched_heightmap"),
    }
    result.recommended_next_action = "review_alignment_debug"
    return result


def _candidate_for_take(data_dir: Path, take_id: str, family: str) -> dict[str, Any]:
    incoming_meta = _read_json(data_dir / "incoming" / take_id / "metadata.json") or {}
    process_entries = process_entries_for_take(data_dir, take_id)
    family_entries = [item for item in process_entries if str(item.get("pipeline_family") or "") == family]
    family_entries = sorted(family_entries, key=lambda item: str(item.get("created_at") or ""), reverse=True)
    successful_entries = [item for item in family_entries if str(item.get("status") or "") in {"success", "warning", "completed"}]

    result_payload = _read_json(data_dir / "processed" / take_id / "result.json")
    result_pipeline = ((result_payload or {}).get("processing_pipeline") or {}) if isinstance(result_payload, dict) else {}
    result_family = str(result_pipeline.get("pipeline_family") or "")
    result_ok = str((result_payload or {}).get("status") or "") in {"success", "warning", "completed"}
    is_native_family_result = family in {"3d", "25d"} and result_family == family and result_ok

    selected_run_id: str | None = None
    if successful_entries:
        selected_run_id = str(successful_entries[0].get("run_id") or "") or None
    elif is_native_family_result:
        selected_run_id = "native_result"
    else:
        result_payload = None

    artifacts = []
    if isinstance(result_payload, dict):
        raw_artifacts = result_payload.get("artifacts")
        artifacts = [item for item in raw_artifacts if isinstance(item, dict)] if isinstance(raw_artifacts, list) else []
    return {
        "take_id": take_id,
        "run_id": selected_run_id,
        "created_at": incoming_meta.get("created_at"),
        "result_payload": result_payload,
        "artifacts": artifacts,
        "family": family,
        "success": bool(result_payload),
    }


def _select_best_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [item for item in items if item.get("success")]
    pool = successful or items
    if not pool:
        return None
    pool = sorted(pool, key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return pool[0]


def _extract_objects(payload: dict[str, Any] | None, kind: str, warnings: list[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        warnings.append(f"Missing {kind} result payload.")
        return []
    raw = payload.get("objects")
    if not isinstance(raw, list):
        warnings.append(f"{kind} result payload has no objects list.")
        return []
    objects: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or item.get("id") or f"{kind}_{idx+1}")
        centroid = item.get("centroid")
        if centroid is None:
            cx = item.get("center_x")
            cy = item.get("center_y")
            if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
                centroid = [float(cx), float(cy)]
        candidate = {
            "object_id": object_id,
            "class_label": item.get("class_label"),
            "class_hint": item.get("class_hint"),
            "confidence": item.get("confidence"),
            "bbox": item.get("bbox"),
            "ellipse": item.get("ellipse"),
            "centroid": centroid,
            "height_metrics": item.get("height_metrics"),
            "footprint_geometry": item.get("footprint_geometry"),
        }
        if candidate["bbox"] is None:
            warnings.append(f"{kind} object '{object_id}' missing bbox.")
        if candidate["centroid"] is None:
            warnings.append(f"{kind} object '{object_id}' missing centroid.")
        objects.append(candidate)
    return objects


def _has_centroid(item: dict[str, Any]) -> bool:
    value = item.get("centroid")
    return isinstance(value, (list, tuple)) and len(value) >= 2


def _has_bbox(item: dict[str, Any]) -> bool:
    value = item.get("bbox")
    return isinstance(value, (list, tuple, dict))


def _next_action_for_status(status: FusionReadinessStatus) -> str:
    if status == "waiting_for_rgb":
        return "acquire_or_process_rgb"
    if status == "waiting_for_heightmap":
        return "acquire_or_process_heightmap"
    if status == "waiting_for_processing":
        return "run_bound_processing"
    if status == "failed_input":
        return "inspect_input_errors"
    return "await_group_completion"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
