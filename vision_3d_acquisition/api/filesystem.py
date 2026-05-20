from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vision_3d_acquisition.api.schemas import RuntimeState, TakeDetail, TakeSummary
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.contracts.artifacts import normalize_processing_artifacts
from vision_3d_acquisition.contracts.execution import build_pipeline_execution_trace
from vision_3d_acquisition.contracts.modalities import assets_by_modality, infer_modalities
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.pipelines.registry import build_stage_outputs_from_artifacts, default_pipeline_info
from vision_3d_acquisition.poc.labels import load_labels
from vision_3d_acquisition.poc.summary import build_poc_run_summary
from vision_3d_acquisition.processing.status_index import process_entries_for_take, summarize_take_processing
from vision_3d_acquisition.state.runtime_status import read_runtime_status
from vision_3d_acquisition.state.sessions import list_sessions, session_summary


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def read_runtime_state(settings: ApiSettings) -> RuntimeState:
    payload = read_runtime_status(settings.state_dir)
    return RuntimeState.model_validate(payload)


def count_take_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir() and not child.name.startswith("."))


def list_take_ids(settings: ApiSettings) -> set[str]:
    take_ids: set[str] = set()
    for root in (settings.incoming_dir, settings.processed_dir):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                take_ids.add(child.name)
    return take_ids


def get_take_summary(settings: ApiSettings, take_id: str) -> TakeSummary:
    incoming_dir = settings.incoming_dir / take_id
    processed_dir = settings.processed_dir / take_id
    metadata = read_json(incoming_dir / "metadata.json") or {}
    result = read_json(processed_dir / "result.json") or {}
    has_ready = (incoming_dir / "READY").is_file()
    has_done = (processed_dir / "DONE").is_file()
    result_status = result.get("status")
    status = "failed" if result_status == "failed" else "processed" if has_done else "incoming"
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    poc_summary = result.get("poc_summary") if isinstance(result.get("poc_summary"), dict) else None
    if result and poc_summary is None:
        try:
            poc_summary = build_poc_run_summary(
                result,
                metadata=metadata,
                output_dir=processed_dir,
                engine=result.get("processing_engine"),
            )
        except ValidationError:
            poc_summary = None
    objects_summary = poc_summary.get("objects") if isinstance(poc_summary, dict) else {}
    plane_summary = poc_summary.get("plane") if isinstance(poc_summary, dict) else {}
    profiling_summary = poc_summary.get("profiling") if isinstance(poc_summary, dict) else {}
    warnings = poc_summary.get("warnings") if isinstance(poc_summary, dict) else []
    modalities = result.get("input_modalities") or infer_modalities(metadata, incoming_dir)
    assets = assets_by_modality(metadata, incoming_dir)
    frameset = metadata.get("frameset") if isinstance(metadata.get("frameset"), dict) else None
    frame_count = (frameset or {}).get("frame_count") or metadata.get("frame_count")
    session_id = metadata.get("session_id") or result.get("session_id")
    frameset_id = (frameset or {}).get("frameset_id") or result.get("frameset_id")
    processing_by_family = summarize_take_processing(
        settings.data_dir,
        take_id,
        has_3d_done=has_done,
        processed_dir=processed_dir,
    )
    dataset_service = DatasetService(settings.data_dir)
    take_management = dataset_service.load_take_metadata(take_id=take_id, source_metadata=metadata)
    dataset_id = take_management.get("dataset_id")
    experiment_session_id = take_management.get("session_id")
    dataset_name = None
    experiment_session_name = None
    if isinstance(dataset_id, str) and dataset_id:
        dataset = dataset_service.get_dataset(dataset_id)
        if isinstance(dataset, dict):
            dataset_name = str(dataset.get("name") or dataset_id)
    if isinstance(dataset_id, str) and dataset_id and isinstance(experiment_session_id, str) and experiment_session_id:
        session = dataset_service.get_session(dataset_id, experiment_session_id)
        if isinstance(session, dict):
            experiment_session_name = str(session.get("name") or experiment_session_id)
    latest_run_status = None
    process_entries = process_entries_for_take(settings.data_dir, take_id)
    if process_entries:
        latest_run_status = str(sorted(process_entries, key=lambda item: str(item.get("created_at") or ""), reverse=True)[0].get("status") or "")
    return TakeSummary(
        take_id=take_id,
        status=status,
        has_ready=has_ready,
        has_done=has_done,
        created_at=metadata.get("created_at") or result.get("processed_at"),
        decision=summary.get("decision"),
        engine=(poc_summary or {}).get("engine") if isinstance(poc_summary, dict) else result.get("processing_engine"),
        object_count=objects_summary.get("found"),
        ball_count=objects_summary.get("estimated_balls"),
        plane_found=plane_summary.get("found"),
        processing_time_ms=profiling_summary.get("total_processing_ms"),
        warning_count=len(warnings) if isinstance(warnings, list) else 0,
        warnings=warnings if isinstance(warnings, list) else [],
        calibration_id=result.get("calibration_id"),
        calibration_file=result.get("calibration_file"),
        calibration_resolution_source=result.get("calibration_resolution_source"),
        modalities=modalities,
        assets=assets,
        frame_count=int(frame_count) if frame_count else None,
        frameset=frameset,
        session_id=session_id,
        frameset_id=frameset_id,
        throughput=result.get("throughput") if isinstance(result.get("throughput"), dict) else None,
        processing_by_family=processing_by_family,
        friendly_name=str(take_management.get("friendly_name") or take_id),
        tags=[str(item) for item in (take_management.get("tags") or []) if str(item)],
        validation_status=str(take_management.get("validation_status") or "unreviewed"),
        expected_class=(str(take_management.get("expected_class")) if take_management.get("expected_class") is not None else None),
        expected_diameter_mm=float(take_management["expected_diameter_mm"]) if isinstance(take_management.get("expected_diameter_mm"), (int, float)) else None,
        thumbnail_path=_take_thumbnail_path(take_id, settings, metadata, result),
        dataset_id=(str(dataset_id) if dataset_id else None),
        dataset_name=dataset_name,
        experiment_session_id=(str(experiment_session_id) if experiment_session_id else None),
        experiment_session_name=experiment_session_name,
        latest_run_status=latest_run_status,
    )


def list_takes(
    settings: ApiSettings,
    session_id: str | None = None,
    dataset_id: str | None = None,
    validation_status: str | None = None,
    tag: str | None = None,
    search: str | None = None,
) -> list[TakeSummary]:
    takes = [get_take_summary(settings, take_id) for take_id in list_take_ids(settings)]
    if session_id:
        takes = [take for take in takes if take.session_id == session_id]
    if dataset_id:
        takes = [take for take in takes if take.dataset_id == dataset_id]
    if validation_status:
        takes = [take for take in takes if (take.validation_status or "").lower() == validation_status.lower()]
    if tag:
        takes = [take for take in takes if tag.lower() in {item.lower() for item in (take.tags or [])}]
    if search:
        q = search.lower()
        takes = [take for take in takes if q in take.take_id.lower() or q in (take.friendly_name or "").lower()]
    return sorted(takes, key=lambda item: item.created_at or item.take_id, reverse=True)


def latest_take(settings: ApiSettings) -> TakeSummary | None:
    processed = [take for take in list_takes(settings) if take.has_done]
    if processed:
        return processed[0]
    incoming = [take for take in list_takes(settings) if take.has_ready]
    return incoming[0] if incoming else None


def get_take_detail(settings: ApiSettings, take_id: str) -> TakeDetail | None:
    if take_id not in list_take_ids(settings):
        return None
    summary = get_take_summary(settings, take_id)
    metadata = read_json(settings.incoming_dir / take_id / "metadata.json")
    result = read_json(settings.processed_dir / take_id / "result.json")
    if not isinstance(result, dict):
        process_result = _process_run_take_result(settings, take_id)
        if isinstance(process_result, dict):
            result = process_result
    if isinstance(result, dict):
        result["artifacts"] = normalize_processing_artifacts(result, output_dir=settings.processed_dir / take_id)
        if not isinstance(result.get("stage_outputs"), list):
            result["stage_outputs"] = build_stage_outputs_from_artifacts(result["artifacts"])
        if not isinstance(result.get("processing_pipeline"), dict):
            result["processing_pipeline"] = default_pipeline_info()
        if not isinstance(result.get("pipeline_execution"), dict):
            result["pipeline_execution"] = build_pipeline_execution_trace(
                pipeline=result.get("processing_pipeline"),
                artifacts=result.get("artifacts") or [],
                profiling=result.get("profiling") if isinstance(result.get("profiling"), dict) else None,
                input_modalities=result.get("input_modalities") if isinstance(result.get("input_modalities"), list) else modalities_from_result(result, metadata, settings, take_id),
                objects=result.get("objects") if isinstance(result.get("objects"), list) else [],
                rejected_objects=result.get("rejected_objects") if isinstance(result.get("rejected_objects"), list) else [],
                result_status=result.get("status"),
                result_error=result.get("error"),
            )
    labels = load_labels(settings.data_dir, take_id)
    modalities = (result or {}).get("input_modalities") or infer_modalities(metadata, settings.incoming_dir / take_id)
    assets = assets_by_modality(metadata, settings.incoming_dir / take_id)
    frameset = metadata.get("frameset") if isinstance(metadata, dict) and isinstance(metadata.get("frameset"), dict) else None
    frame_count = (frameset or {}).get("frame_count") or (metadata or {}).get("frame_count")
    session_id = (metadata or {}).get("session_id") or (result or {}).get("session_id")
    frameset_id = (frameset or {}).get("frameset_id") or (result or {}).get("frameset_id")
    dataset_service = DatasetService(settings.data_dir)
    take_management = dataset_service.load_take_metadata(take_id=take_id, source_metadata=metadata)
    object_annotations_raw = take_management.get("object_annotations")
    object_annotations = [item for item in object_annotations_raw if isinstance(item, dict)] if isinstance(object_annotations_raw, list) else []
    candidates = _extract_take_candidates(result)
    matched_annotations = dataset_service.match_object_annotations(annotations=object_annotations, candidates=candidates)
    return TakeDetail(
        take_id=summary.take_id,
        status=summary.status,
        has_ready=summary.has_ready,
        has_done=summary.has_done,
        created_at=summary.created_at,
        metadata=metadata,
        result=result,
        labels=labels,
        modalities=modalities,
        assets=assets,
        frame_count=int(frame_count) if frame_count else None,
        frameset=frameset,
        session_id=session_id,
        frameset_id=frameset_id,
        throughput=(result or {}).get("throughput") if isinstance((result or {}).get("throughput"), dict) else None,
        processing_by_family=summary.processing_by_family,
        take_metadata=take_management,
        run_history=_take_run_history(settings, take_id, result),
        object_annotations=matched_annotations,
    )


def _process_run_take_result(settings: ApiSettings, take_id: str) -> dict[str, Any] | None:
    entries = process_entries_for_take(settings.data_dir, take_id)
    if not entries:
        return None
    entries = sorted(entries, key=lambda item: str(item.get("created_at") or ""), reverse=True)
    latest = next((item for item in entries if str(item.get("pipeline_family") or "") == "2d"), entries[0])
    instance_id = str(latest.get("pipeline_instance_id") or "")
    run_id = str(latest.get("run_id") or "")
    if not instance_id or not run_id:
        return None
    instance_path = settings.data_dir / "processes" / "instances" / f"{instance_id}.json"
    if not instance_path.is_file():
        return None
    instance_payload = read_json(instance_path)
    if not isinstance(instance_payload, dict):
        return None
    history = instance_payload.get("execution_history")
    if not isinstance(history, list):
        return None
    run = next((item for item in history if isinstance(item, dict) and str(item.get("run_id") or "") == run_id), None)
    if not isinstance(run, dict):
        return None

    artifacts_raw = run.get("artifacts")
    artifacts: list[dict[str, Any]] = []
    if isinstance(artifacts_raw, list):
        for item in artifacts_raw:
            if not isinstance(item, dict):
                continue
            artifact = dict(item)
            original_stage = str(artifact.get("stage_id") or "")
            mapped_stage = _map_stage_alias(original_stage)
            if mapped_stage:
                artifact["stage_id"] = mapped_stage
            metadata = dict(artifact.get("metadata") or {})
            metadata.setdefault("original_step_id", original_stage or None)
            metadata["run_id"] = run_id
            metadata["pipeline_instance_id"] = instance_id
            artifact["metadata"] = metadata
            artifacts.append(artifact)

    result: dict[str, Any] = {
        "take_id": take_id,
        "processed_at": str(run.get("timestamp") or latest.get("created_at") or ""),
        "processing_mode": "real",
        "processing_engine": "process_service",
        "input_modalities": ["rgb"],
        "algorithm_stage": "classification",
        "status": "ok" if str(run.get("status") or "") in {"success", "warning", "completed"} else "failed",
        "summary": {
            "object_count": int((run.get("summary") or {}).get("total_objects") or 0) if isinstance(run.get("summary"), dict) else 0,
            "ball_count": int((run.get("summary") or {}).get("balls") or 0) if isinstance(run.get("summary"), dict) else 0,
            "non_ball_count": int((run.get("summary") or {}).get("non_balls") or 0) if isinstance(run.get("summary"), dict) else 0,
            "decision": "review",
            "confidence": None,
        },
        "objects": [],
        "rejected_objects": [],
        "files": {},
        "timing_ms": {"load": 0.0, "segmentation": 0.0, "classification": 0.0, "total": 0.0},
        "error": None if str(run.get("status") or "") in {"success", "warning", "completed"} else "process_service_run_failed",
        "artifacts": artifacts,
        "processing_pipeline": default_pipeline_info("mining_steel_ball_classification_2d"),
    }
    step_reports = run.get("step_reports")
    if isinstance(step_reports, list):
        total = 0.0
        profile_stages: list[dict[str, Any]] = []
        for report in step_reports:
            if not isinstance(report, dict):
                continue
            duration = float(report.get("duration_ms") or 0.0)
            total += duration
            stage = str(report.get("step_id") or "")
            profile_stages.append(
                {
                    "name": stage,
                    "category": _map_stage_alias(stage) or stage,
                    "duration_ms": duration,
                }
            )
        result["timing_ms"] = {"load": 0.0, "segmentation": 0.0, "classification": total, "total": total}
        result["profiling"] = {
            "total_ms": total,
            "production_ms": total,
            "debug_artifacts_ms": 0.0,
            "io_ms": 0.0,
            "stages": profile_stages,
        }
    result["pipeline_execution"] = build_pipeline_execution_trace(
        pipeline=result.get("processing_pipeline"),
        artifacts=artifacts,
        profiling=result.get("profiling") if isinstance(result.get("profiling"), dict) else None,
        input_modalities=["rgb"],
        objects=[],
        rejected_objects=[],
        result_status=result.get("status"),
        result_error=result.get("error"),
    )
    result["stage_outputs"] = build_stage_outputs_from_artifacts(artifacts)
    return result


def _map_stage_alias(stage_id: str) -> str:
    if stage_id in {"threshold", "morphology", "segmentation"}:
        return "segmentation"
    if stage_id in {"blob_detection", "detection"}:
        return "detection"
    if stage_id in {"ellipse_fitting", "ellipse", "geometry_metrics"}:
        return "ellipse_fitting"
    if stage_id in {"metrics", "measurement"}:
        return "measurement"
    if stage_id in {"classification", "overlay", "summary"}:
        return "classification"
    if stage_id in {"input", "crop_roi", "rgb_to_gray", "normalize_lighting"}:
        return "input"
    return stage_id


def list_session_summaries(settings: ApiSettings) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for session in list_sessions(settings.sessions_dir):
        summary = session_summary(settings.sessions_dir, session.session_id)
        if summary is not None:
            summaries.append(summary)
    return summaries


def get_session_summary(settings: ApiSettings, session_id: str) -> dict[str, Any] | None:
    summary = session_summary(settings.sessions_dir, session_id)
    if summary is None:
        return None
    takes = list_takes(settings, session_id=session_id)
    return {
        **summary,
        "takes": [take.model_dump(mode="json") for take in takes],
        "processed_count": sum(1 for take in takes if take.has_done),
        "failed_count": sum(1 for take in takes if take.status == "failed"),
    }


def safe_take_file(settings: ApiSettings, take_id: str, filename: str) -> Path | None:
    if filename in {"", ".", ".."}:
        return None
    normalized = filename.replace("\\", "/").strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        return None
    for base in (settings.processed_dir / take_id, settings.incoming_dir / take_id):
        candidate = (base / normalized).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            return None
        if candidate.is_file():
            return candidate
    shared_candidate = (settings.data_dir / normalized).resolve()
    try:
        shared_candidate.relative_to((settings.data_dir / "processes" / "runs").resolve())
    except ValueError:
        return None
    if not shared_candidate.is_file():
        return None
    entries = process_entries_for_take(settings.data_dir, take_id)
    allowed_roots: list[Path] = []
    for entry in entries:
        path = entry.get("path")
        if not path:
            continue
        try:
            allowed_roots.append(Path(str(path)).resolve())
        except Exception:
            continue
    for root in allowed_roots:
        try:
            shared_candidate.relative_to(root)
            return shared_candidate
        except ValueError:
            continue
    return None


def modalities_from_result(
    result: dict[str, Any],
    metadata: dict[str, Any] | None,
    settings: ApiSettings,
    take_id: str,
) -> list[str]:
    return result.get("input_modalities") or infer_modalities(metadata, settings.incoming_dir / take_id)


def _take_thumbnail_path(
    take_id: str,
    settings: ApiSettings,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    assets = assets_by_modality(metadata, settings.incoming_dir / take_id)
    for modality in ("rgb", "reflectance", "laser_rgb"):
        group = assets.get(modality)
        if isinstance(group, dict):
            for filename in group.values():
                if filename:
                    return str(filename)
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    for key in ("overlay", "debug_segmentation", "debug_plane_segmentation"):
        candidate = files.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _take_run_history(settings: ApiSettings, take_id: str, result: dict[str, Any] | None) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    if isinstance(result, dict):
        history.append(
            {
                "run_id": "3d_result",
                "pipeline": str((result.get("processing_pipeline") or {}).get("id") or "3d_ball_inspection"),
                "timestamp": result.get("processed_at"),
                "status": result.get("status"),
                "duration_ms": (result.get("timing_ms") or {}).get("total"),
                "backend": "legacy",
            }
        )
    for entry in sorted(process_entries_for_take(settings.data_dir, take_id), key=lambda item: str(item.get("created_at") or ""), reverse=True):
        history.append(
            {
                "run_id": entry.get("run_id"),
                "pipeline": entry.get("pipeline_family"),
                "timestamp": entry.get("created_at"),
                "status": entry.get("status"),
                "duration_ms": None,
                "backend": "process_service",
                "pipeline_instance_id": entry.get("pipeline_instance_id"),
            }
        )
    return history


def _extract_take_candidates(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    rows: list[dict[str, Any]] = []
    blob_metrics = next((item for item in artifacts if isinstance(item, dict) and str(item.get("artifact_id") or "") in {"blob_metrics", "detection_metrics"}), None)
    if isinstance(blob_metrics, dict):
        metadata = blob_metrics.get("metadata")
        entries = (metadata or {}).get("entries") if isinstance(metadata, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                candidate_id = str(entry.get("blob_id") or entry.get("object_id") or entry.get("id") or "")
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "source_stage": "blob_detection",
                        "source_artifact_id": "blob_metrics",
                        "bbox": entry.get("bbox"),
                        "centroid": entry.get("centroid"),
                    }
                )
    ellipse_metrics = next((item for item in artifacts if isinstance(item, dict) and str(item.get("artifact_id") or "") in {"ellipse_metrics", "geometry_metrics"}), None)
    if isinstance(ellipse_metrics, dict):
        metadata = ellipse_metrics.get("metadata")
        entries = (metadata or {}).get("entries") if isinstance(metadata, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                candidate_id = str(entry.get("candidate_id") or entry.get("object_id") or entry.get("id") or "")
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "source_stage": "ellipse_fitting",
                        "source_artifact_id": "ellipse_metrics",
                        "bbox": entry.get("bbox"),
                        "centroid": entry.get("centroid"),
                    }
                )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("candidate_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique
