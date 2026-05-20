from __future__ import annotations

from typing import Any


def build_pipeline_execution_trace(
    *,
    pipeline: dict[str, Any] | None,
    artifacts: list[dict[str, Any]],
    profiling: dict[str, Any] | None,
    input_modalities: list[str] | None,
    objects: list[dict[str, Any]],
    rejected_objects: list[dict[str, Any]],
    result_status: str | None,
    result_error: str | None,
) -> dict[str, Any]:
    stages = list((pipeline or {}).get("stages") or [])
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for item in artifacts:
        stage_id = str(item.get("stage_id") or "result")
        by_stage.setdefault(stage_id, []).append(item)
    profile_stages = list((profiling or {}).get("stages") or [])
    reports: list[dict[str, Any]] = []
    previous_outputs: list[str] = []
    available_modalities = set(input_modalities or [])
    for index, stage in enumerate(stages):
        stage_id = str(stage.get("id") or stage.get("stage_id") or f"stage_{index}")
        required_modalities = list(stage.get("required_modalities") or [])
        output_ids = [str(item.get("artifact_id")) for item in by_stage.get(stage_id, []) if item.get("artifact_id")]
        matches = [item for item in profile_stages if _profile_matches_stage(item, stage_id)]
        durations = [float(item.get("duration_ms") or 0.0) for item in matches]
        started = [item.get("started_at") for item in matches if isinstance(item.get("started_at"), (int, float))]
        finished = [item.get("ended_at") for item in matches if isinstance(item.get("ended_at"), (int, float))]
        warnings = []
        errors = []
        missing = [modality for modality in required_modalities if modality not in available_modalities]
        if missing:
            warnings.append(f"Skipped: missing modalities ({', '.join(missing)}).")
        if stage.get("implemented") is False:
            warnings.append("Skipped: stage is not implemented in this pipeline.")
        if result_status == "failed" and result_error:
            if stage_id in {"classification", "measurement"} or _is_likely_active_stage(stage_id, output_ids, matches):
                errors.append(result_error)
        status = _stage_status(
            has_outputs=bool(output_ids),
            has_profile=bool(matches),
            warnings=warnings,
            errors=errors,
        )
        reports.append(
            {
                "stage_id": stage_id,
                "status": status,
                "started_at": min(started) if started else None,
                "finished_at": max(finished) if finished else None,
                "duration_ms": round(sum(durations), 3),
                "warnings": warnings,
                "errors": errors,
                "input_artifact_ids": list(previous_outputs),
                "output_artifact_ids": output_ids,
                "object_count": len(objects) if stage_id in {"segmentation", "classification", "measurement"} else 0,
                "rejected_count": len(rejected_objects) if stage_id in {"classification", "measurement"} else 0,
            }
        )
        if output_ids:
            previous_outputs = output_ids
    return {
        "pipeline_id": (pipeline or {}).get("id"),
        "stages": reports,
    }


def _stage_status(*, has_outputs: bool, has_profile: bool, warnings: list[str], errors: list[str]) -> str:
    if errors:
        return "failed"
    if warnings:
        return "warning" if has_outputs or has_profile else "skipped"
    if has_outputs or has_profile:
        return "success"
    return "skipped"


def _profile_matches_stage(profile_stage: dict[str, Any], stage_id: str) -> bool:
    name = str(profile_stage.get("name") or "").lower()
    category = str(profile_stage.get("category") or "").lower()
    target = stage_id.lower()
    aliases = {
        "segmentation": ("segmentation", "plane", "foreground", "cluster", "preprocess", "calibration_filtering"),
        "classification": ("classification", "classify", "statistics"),
        "measurement": ("measurement", "measure", "sphere", "ellipse"),
    }
    candidates = aliases.get(target, (target,))
    return any(item in name or item in category for item in candidates)


def _is_likely_active_stage(stage_id: str, output_ids: list[str], matches: list[dict[str, Any]]) -> bool:
    if output_ids or matches:
        return True
    return stage_id in {"classification", "measurement"}
