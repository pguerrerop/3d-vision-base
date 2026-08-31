from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vision_3d_acquisition.api.schemas import RuntimeState, TakeDetail, TakeSummary
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.acquisition.processing_status import read_status as read_acquisition_processing_status
from vision_3d_acquisition.contracts.artifacts import normalize_processing_artifacts
from vision_3d_acquisition.contracts.execution import build_pipeline_execution_trace
from vision_3d_acquisition.contracts.object_candidates import normalize_object_candidates
from vision_3d_acquisition.contracts.modalities import assets_by_modality, infer_modalities
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.pipelines.registry import build_stage_outputs_from_artifacts, default_pipeline_info
from vision_3d_acquisition.pipelines.processing_units import (
    PROCESSING_UNIT_REGISTRY_VERSION,
    build_processing_unit_trace,
    build_recipe_snapshot,
    processing_unit_contract_fingerprint,
)
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


def _processed_labels_from_result(result: dict[str, Any]) -> tuple[str | None, str | None]:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    classification = result.get("classification") if isinstance(result.get("classification"), dict) else {}
    objects = result.get("objects") if isinstance(result.get("objects"), list) else []
    first_object = objects[0] if objects and isinstance(objects[0], dict) else {}
    class_label = next(
        (
            str(value).strip()
            for value in (
                classification.get("final_class_label"),
                classification.get("class_label"),
                classification.get("label"),
                summary.get("class_label"),
                summary.get("label"),
                first_object.get("class_label"),
                first_object.get("label"),
            )
            if value is not None and str(value).strip()
        ),
        None,
    )
    superclass = next(
        (
            str(value).strip()
            for value in (
                classification.get("superclass"),
                summary.get("superclass"),
                first_object.get("superclass"),
            )
            if value is not None and str(value).strip()
        ),
        None,
    )
    return class_label, superclass


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


def get_take_summary(
    settings: ApiSettings,
    take_id: str,
    *,
    include_acquisition_processing_status: bool = True,
) -> TakeSummary:
    """Build a take's summary.

    ``include_acquisition_processing_status`` exists because that one field is
    the whole payload: it embeds the run's complete result.json, 159 KB at the
    median and up to 453 KB, against ~1.6 KB for everything else. TakeDetail
    needs it and asks for it; the listing does not read it and does not, unless
    a caller explicitly opts in.
    """
    incoming_dir = settings.incoming_dir / take_id
    processed_dir = settings.processed_dir / take_id
    metadata = read_json(incoming_dir / "metadata.json") or {}
    result = read_json(processed_dir / "result.json") or {}
    has_ready = (incoming_dir / "READY").is_file()
    has_done = (processed_dir / "DONE").is_file()
    result_status = result.get("status")
    status = "failed" if result_status == "failed" else "processed" if has_done else "incoming"
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    processed_class_label, processed_superclass = _processed_labels_from_result(result) if result else (None, None)
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
    modality_labels = _modality_labels(modalities, metadata)
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
    experiment_session_type = None
    if isinstance(dataset_id, str) and dataset_id:
        dataset = dataset_service.get_dataset(dataset_id)
        if isinstance(dataset, dict):
            dataset_name = str(dataset.get("name") or dataset_id)
    if isinstance(dataset_id, str) and dataset_id and isinstance(experiment_session_id, str) and experiment_session_id:
        session = dataset_service.get_session(dataset_id, experiment_session_id)
        if isinstance(session, dict):
            experiment_session_name = str(session.get("name") or experiment_session_id)
            experiment_session_type = str(session.get("session_type") or "engineering")
    latest_run_status = None
    process_entries = process_entries_for_take(settings.data_dir, take_id)
    if process_entries:
        latest_run_status = str(sorted(process_entries, key=lambda item: str(item.get("created_at") or ""), reverse=True)[0].get("status") or "")
    acquisition_processing = (
        read_acquisition_processing_status(settings.data_dir, take_id)
        if include_acquisition_processing_status
        else None
    )
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
        modality_labels=modality_labels,
        assets=assets,
        frame_count=int(frame_count) if frame_count else None,
        frameset=frameset,
        session_id=session_id,
        frameset_id=frameset_id,
        throughput=result.get("throughput") if isinstance(result.get("throughput"), dict) else None,
        processing_by_family=processing_by_family,
        friendly_name=str(take_management.get("friendly_name") or take_id),
        tags=[str(item) for item in (take_management.get("tags") or []) if str(item)],
        semantic_labels=[str(item) for item in (take_management.get("semantic_labels") or []) if str(item)],
        superclass_labels=[str(item) for item in (take_management.get("superclass_labels") or []) if str(item)],
        processed_class_label=processed_class_label,
        processed_superclass=processed_superclass,
        normalized_class=(str(take_management.get("normalized_class")) if take_management.get("normalized_class") is not None else None),
        normalization_version=(str(take_management.get("normalization_version")) if take_management.get("normalization_version") is not None else None),
        validation_status=str(take_management.get("validation_status") or "unreviewed"),
        expected_class=(str(take_management.get("expected_class")) if take_management.get("expected_class") is not None else None),
        expected_diameter_mm=float(take_management["expected_diameter_mm"]) if isinstance(take_management.get("expected_diameter_mm"), (int, float)) else None,
        physical_object_id=(str(take_management.get("physical_object_id")) if take_management.get("physical_object_id") is not None else None),
        thumbnail_path=_take_thumbnail_path(take_id, settings, metadata, result),
        dataset_id=(str(dataset_id) if dataset_id else None),
        dataset_name=dataset_name,
        experiment_session_id=(str(experiment_session_id) if experiment_session_id else None),
        experiment_session_name=experiment_session_name,
        experiment_session_type=experiment_session_type,
        latest_run_status=latest_run_status,
        archived=bool(take_management.get("archived")),
        categories=[str(item) for item in (take_management.get("categories") or []) if str(item)],
        reference_type=(str(take_management.get("reference_type")) if take_management.get("reference_type") is not None else None),
        is_reference=bool(take_management.get("is_reference")),
        is_golden_sample=bool(take_management.get("is_golden_sample")),
        acquisition_processing_status=acquisition_processing,
    )


def _catalog(settings: ApiSettings):
    """The catalog connection, or None when reads should walk the filesystem.

    Stage 1 serves the listing from data/index.db. SENSOR_STUDIO_INDEX=off falls
    back to the filesystem scan below, which stays in place for one release so a
    problem in the index is a restart away from being worked around rather than a
    revert. An index that has never completed a full scan, or whose take count no
    longer matches the disk, is not trusted — see catalog_ready_for_reads.
    """
    from vision_3d_acquisition.storage import catalog_sync

    if not catalog_sync.catalog_available(settings.data_dir):
        return None
    return catalog_sync.catalog_ready_for_reads(settings.data_dir, len(list_take_ids(settings)))


def list_takes(
    settings: ApiSettings,
    session_id: str | None = None,
    dataset_id: str | None = None,
    validation_status: str | None = None,
    tag: str | None = None,
    semantic_label: str | None = None,
    superclass_label: str | None = None,
    search: str | None = None,
    physical_object_id: str | None = None,
    session_type: str | None = None,
    category: str | None = None,
    reference_type: str | None = None,
    is_reference: bool | None = None,
    is_golden_sample: bool | None = None,
    include_archived: bool = False,
    include_acquisition_processing_status: bool = False,
) -> list[TakeSummary]:
    conn = _catalog(settings)
    if conn is not None:
        from vision_3d_acquisition.storage import catalog_queries

        return catalog_queries.list_takes(
            conn,
            {
                "settings": settings,
                "include_acquisition_processing_status": include_acquisition_processing_status,
                "session_id": session_id,
                "dataset_id": dataset_id,
                "validation_status": validation_status,
                "tag": tag,
                "semantic_label": semantic_label,
                "superclass_label": superclass_label,
                "search": search,
                "physical_object_id": physical_object_id,
                "session_type": session_type,
                "category": category,
                "reference_type": reference_type,
                "is_reference": is_reference,
                "is_golden_sample": is_golden_sample,
                "include_archived": include_archived,
            },
        )

    takes = [
        get_take_summary(
            settings,
            take_id,
            include_acquisition_processing_status=include_acquisition_processing_status,
        )
        for take_id in list_take_ids(settings)
    ]
    if session_id:
        takes = [take for take in takes if take.session_id == session_id]
    if dataset_id:
        takes = [take for take in takes if take.dataset_id == dataset_id]
    if validation_status:
        takes = [take for take in takes if (take.validation_status or "").lower() == validation_status.lower()]
    if tag:
        takes = [take for take in takes if tag.lower() in {item.lower() for item in (take.tags or [])}]
    if semantic_label:
        takes = [take for take in takes if semantic_label.lower() in {item.lower() for item in (take.semantic_labels or [])}]
    if superclass_label:
        takes = [take for take in takes if superclass_label.lower() in {item.lower() for item in (take.superclass_labels or [])}]
    if search:
        q = search.lower()
        takes = [take for take in takes if q in take.take_id.lower() or q in (take.friendly_name or "").lower()]
    if physical_object_id:
        q = physical_object_id.lower().strip()
        takes = [take for take in takes if (take.physical_object_id or "").lower() == q]
    if session_type:
        q = session_type.lower().strip()
        takes = [take for take in takes if (take.experiment_session_type or "").lower() == q]
    if category:
        q = category.lower().strip()
        takes = [take for take in takes if q in {item.lower() for item in (take.categories or [])}]
    if reference_type:
        q = reference_type.lower().strip()
        takes = [take for take in takes if (take.reference_type or "").lower() == q]
    if is_reference is not None:
        takes = [take for take in takes if bool(take.is_reference) is bool(is_reference)]
    if is_golden_sample is not None:
        takes = [take for take in takes if bool(take.is_golden_sample) is bool(is_golden_sample)]
    if not include_archived:
        dataset_service = DatasetService(settings.data_dir)
        visible: list[TakeSummary] = []
        for take in takes:
            meta = dataset_service.load_take_metadata(take_id=take.take_id, source_metadata={})
            if bool(meta.get("archived")):
                continue
            visible.append(take)
        takes = visible
    return sorted(takes, key=lambda item: item.created_at or item.take_id, reverse=True)


def list_takes_paged(
    settings: ApiSettings,
    *,
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
    dataset_id: str | None = None,
    validation_status: str | None = None,
    tag: str | None = None,
    semantic_label: str | None = None,
    superclass_label: str | None = None,
    search: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    expected_class: str | None = None,
    split: str | None = None,
    calibration_linkage_only: bool = False,
    physical_object_id: str | None = None,
    session_type: str | None = None,
    category: str | None = None,
    reference_type: str | None = None,
    is_reference: bool | None = None,
    is_golden_sample: bool | None = None,
    include_archived: bool = False,
    include_acquisition_processing_status: bool = False,
    profile: bool = False,
) -> dict[str, Any]:
    conn = _catalog(settings)
    if conn is not None:
        from vision_3d_acquisition.storage import catalog_queries

        return catalog_queries.list_takes_paged(
            conn,
            {
                "settings": settings,
                "include_acquisition_processing_status": include_acquisition_processing_status,
                "limit": limit,
                "offset": offset,
                "session_id": session_id,
                "dataset_id": dataset_id,
                "validation_status": validation_status,
                "tag": tag,
                "semantic_label": semantic_label,
                "superclass_label": superclass_label,
                "search": search,
                "created_from": created_from,
                "created_to": created_to,
                "expected_class": expected_class,
                "split": split,
                "calibration_linkage_only": calibration_linkage_only,
                "physical_object_id": physical_object_id,
                "session_type": session_type,
                "category": category,
                "reference_type": reference_type,
                "is_reference": is_reference,
                "is_golden_sample": is_golden_sample,
                "include_archived": include_archived,
                "profile": profile,
            },
        )

    total_started_at = time.perf_counter()
    resolved_limit = max(1, min(int(limit), 200))
    resolved_offset = max(0, int(offset))
    dataset_service = DatasetService(settings.data_dir)
    session_type_cache: dict[tuple[str, str], str] = {}

    created_from_iso = (created_from or "").strip()
    created_to_iso = (created_to or "").strip()

    def _match_common_filters(
        take_id: str,
        metadata: dict[str, Any],
        take_meta: dict[str, Any],
        *,
        include_dataset_session_filters: bool,
    ) -> bool:
        take_dataset_id = str(take_meta.get("dataset_id") or "")
        take_session_id = str(take_meta.get("session_id") or metadata.get("session_id") or "")
        take_session_type = ""
        if take_dataset_id and take_session_id:
            key = (take_dataset_id, take_session_id)
            if key not in session_type_cache:
                session_payload = dataset_service.get_session(take_dataset_id, take_session_id)
                session_type_cache[key] = str((session_payload or {}).get("session_type") or "")
            take_session_type = session_type_cache[key]

        if include_dataset_session_filters:
            if session_id and take_session_id != session_id:
                return False
            if dataset_id and take_dataset_id != dataset_id:
                return False

        if validation_status and str(take_meta.get("validation_status") or "unreviewed").lower() != validation_status.lower():
            return False
        if tag and tag.lower() not in {str(item).lower() for item in (take_meta.get("tags") or [])}:
            return False
        if semantic_label and semantic_label.lower() not in {str(item).lower() for item in (take_meta.get("semantic_labels") or [])}:
            return False
        if superclass_label and superclass_label.lower() not in {str(item).lower() for item in (take_meta.get("superclass_labels") or [])}:
            return False
        if physical_object_id and str(take_meta.get("physical_object_id") or "").lower().strip() != physical_object_id.lower().strip():
            return False
        if session_type and take_session_type.lower() != session_type.lower().strip():
            return False
        if category and category.lower().strip() not in {str(item).lower() for item in (take_meta.get("categories") or [])}:
            return False
        if reference_type and str(take_meta.get("reference_type") or "").lower().strip() != reference_type.lower().strip():
            return False
        if is_reference is not None and bool(take_meta.get("is_reference")) is not bool(is_reference):
            return False
        if is_golden_sample is not None and bool(take_meta.get("is_golden_sample")) is not bool(is_golden_sample):
            return False
        if search:
            q = search.lower()
            friendly = str(take_meta.get("friendly_name") or "")
            if q not in take_id.lower() and q not in friendly.lower():
                return False
        if expected_class:
            q = expected_class.lower().strip()
            value = str(take_meta.get("expected_class") or "").lower()
            if q not in value:
                return False
        if split:
            q = split.lower().strip()
            split_value = str(take_meta.get("split") or "").lower().strip()
            if split_value != q:
                return False
        if calibration_linkage_only:
            calibration_id = str((take_meta.get("calibration_id") or metadata.get("calibration_id") or "")).strip()
            if not calibration_id:
                return False

        created_at = str(metadata.get("created_at") or take_meta.get("created_at") or "")
        if created_from_iso and created_at and created_at[:10] < created_from_iso:
            return False
        if created_to_iso and created_at and created_at[:10] > created_to_iso:
            return False
        if (created_from_iso or created_to_iso) and not created_at:
            return False
        return True

    candidates: list[tuple[str, str]] = []
    total_count = 0
    summary_counts = {
        "validation": {
            "validated": 0,
            "unreviewed": 0,
            "rejected": 0,
            "needs_review": 0,
            "golden_sample": 0,
            "benchmark_approved": 0,
        },
        "missing_labels": 0,
        "missing_split": 0,
        "missing_calibration": 0,
        "processing_failed": 0,
        "processing_incomplete": 0,
        "no_objects_detected": 0,
    }
    list_take_ids_started_at = time.perf_counter()
    take_ids = list_take_ids(settings)
    list_take_ids_ms = (time.perf_counter() - list_take_ids_started_at) * 1000.0

    scan_started_at = time.perf_counter()
    for take_id in take_ids:
        metadata = read_json(settings.incoming_dir / take_id / "metadata.json") or {}
        take_meta = dataset_service.load_take_metadata(take_id=take_id, source_metadata=metadata)

        if not include_archived and bool(take_meta.get("archived")):
            continue

        if dataset_id and str(take_meta.get("dataset_id") or "") != dataset_id:
            continue

        total_count += 1

        if not _match_common_filters(take_id, metadata, take_meta, include_dataset_session_filters=True):
            continue

        validation_state = str(take_meta.get("validation_status") or "unreviewed").lower().strip()
        validation_map = {
            "valid": "validated",
            "validated": "validated",
            "invalid": "rejected",
            "rejected": "rejected",
            "needs_review": "needs_review",
            "golden_sample": "golden_sample",
            "benchmark_approved": "benchmark_approved",
            "unreviewed": "unreviewed",
        }
        key = validation_map.get(validation_state, "unreviewed")
        summary_counts["validation"][key] = int(summary_counts["validation"][key]) + 1

        expected_class_value = str(take_meta.get("expected_class") or "").strip()
        if not expected_class_value:
            summary_counts["missing_labels"] = int(summary_counts["missing_labels"]) + 1

        split_value = str(take_meta.get("split") or "").strip()
        if not split_value:
            summary_counts["missing_split"] = int(summary_counts["missing_split"]) + 1

        calibration_value = str((take_meta.get("calibration_id") or metadata.get("calibration_id") or "")).strip()
        if not calibration_value:
            summary_counts["missing_calibration"] = int(summary_counts["missing_calibration"]) + 1

        latest_status = str(take_meta.get("latest_run_status") or "").lower().strip()
        if latest_status == "failed":
            summary_counts["processing_failed"] = int(summary_counts["processing_failed"]) + 1
        if latest_status not in {"success", "completed", "done"}:
            summary_counts["processing_incomplete"] = int(summary_counts["processing_incomplete"]) + 1

        obj_count = take_meta.get("object_count")
        if isinstance(obj_count, (int, float)) and int(obj_count) <= 0:
            summary_counts["no_objects_detected"] = int(summary_counts["no_objects_detected"]) + 1

        created_at = str(metadata.get("created_at") or take_meta.get("created_at") or "")
        candidates.append((created_at, take_id))
    scan_and_filter_ms = (time.perf_counter() - scan_started_at) * 1000.0

    sort_started_at = time.perf_counter()
    candidates.sort(key=lambda item: item[0] or item[1], reverse=True)
    sort_candidates_ms = (time.perf_counter() - sort_started_at) * 1000.0
    page_take_ids = [take_id for _created_at, take_id in candidates[resolved_offset : resolved_offset + resolved_limit]]

    hydrate_started_at = time.perf_counter()
    page_items = [
        get_take_summary(
            settings,
            take_id,
            include_acquisition_processing_status=include_acquisition_processing_status,
        )
        for take_id in page_take_ids
    ]
    hydrate_page_items_ms = (time.perf_counter() - hydrate_started_at) * 1000.0
    next_offset = resolved_offset + len(page_items)
    has_more = next_offset < len(candidates)
    payload = {
        "items": page_items,
        "limit": resolved_limit,
        "offset": resolved_offset,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "filtered_count": len(candidates),
        "total_count": total_count,
        "summary_counts": summary_counts,
    }
    if profile:
        payload["profile"] = {
            "enabled": True,
            "limit": resolved_limit,
            "offset": resolved_offset,
            "counters": {
                "scanned_take_ids": len(take_ids),
                "candidate_count": len(candidates),
                "page_item_count": len(page_items),
                "total_count": total_count,
            },
            "phase_ms": {
                "list_take_ids": round(list_take_ids_ms, 2),
                "scan_and_filter": round(scan_and_filter_ms, 2),
                "sort_candidates": round(sort_candidates_ms, 2),
                "hydrate_page_items": round(hydrate_page_items_ms, 2),
                "total": round((time.perf_counter() - total_started_at) * 1000.0, 2),
            },
        }
    return payload


def latest_take(settings: ApiSettings) -> TakeSummary | None:
    processed = [take for take in list_takes(settings) if take.has_done]
    if processed:
        return processed[0]
    incoming = [take for take in list_takes(settings) if take.has_ready]
    return incoming[0] if incoming else None


def get_take_detail(settings: ApiSettings, take_id: str, *, run_id: str | None = None) -> TakeDetail | None:
    if take_id not in list_take_ids(settings):
        return None
    summary = get_take_summary(settings, take_id)
    metadata = read_json(settings.incoming_dir / take_id / "metadata.json")
    result: dict[str, Any] | None
    output_dir = settings.processed_dir / take_id
    if run_id and run_id != "latest":
        resolved = _resolve_specific_run(settings, take_id, run_id)
        if resolved is None:
            return None
        result, output_dir = resolved
    else:
        result = read_json(settings.processed_dir / take_id / "result.json")
        if not isinstance(result, dict):
            process_result = _process_run_take_result(settings, take_id)
            if isinstance(process_result, dict):
                result = process_result
    if isinstance(result, dict):
        result["artifacts"] = normalize_processing_artifacts(result, output_dir=output_dir)
        if not isinstance(result.get("stage_outputs"), list):
            result["stage_outputs"] = build_stage_outputs_from_artifacts(result["artifacts"])
        if not isinstance(result.get("processing_pipeline"), dict):
            result["processing_pipeline"] = default_pipeline_info()
        pipeline_info = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
        pipeline_id = str(pipeline_info.get("id") or pipeline_info.get("name") or default_pipeline_info().get("id"))
        if not isinstance(pipeline_info.get("processing_units"), list):
            try:
                result["processing_pipeline"] = default_pipeline_info(pipeline_id)
            except KeyError:
                pass
            else:
                pipeline_info = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
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
        if pipeline_id == "mining_steel_ball_classification_25d":
            processing_units = list(pipeline_info.get("processing_units") or [])
            stage_params = result.get("stage_params") if isinstance(result.get("stage_params"), dict) else {}
            if not isinstance(result.get("recipe_snapshot"), dict):
                result["recipe_snapshot"] = build_recipe_snapshot(
                    pipeline_id=pipeline_id,
                    pipeline_version=str(pipeline_info.get("version") or "1.0"),
                    registry_version=PROCESSING_UNIT_REGISTRY_VERSION,
                    contract_fingerprint=processing_unit_contract_fingerprint(processing_units),
                    units=processing_units,
                    stage_params=stage_params,
                    recipe_version_id=result.get("recipe_version_id") if isinstance(result.get("recipe_version_id"), str) else None,
                    enabled_units=[str(item.get("id")) for item in processing_units if isinstance(item, dict) and item.get("id")],
                    provenance={"source": "filesystem_backfill"},
                    artifact_policy=result.get("artifact_policy") if isinstance(result.get("artifact_policy"), dict) else None,
                    calibration_snapshot_reference=(
                        result.get("calibration_snapshot_reference")
                        if isinstance(result.get("calibration_snapshot_reference"), dict)
                        else None
                    ),
                )
            result["processing_unit_trace"] = build_processing_unit_trace(
                pipeline_id=pipeline_id,
                units=processing_units,
                artifacts=list(result.get("artifacts") or []),
                stage_params=stage_params,
                runtime_trace=result.get("processing_unit_trace") if isinstance(result.get("processing_unit_trace"), dict) else None,
            )
        result["object_candidates"] = normalize_object_candidates(result)
    labels = load_labels(settings.data_dir, take_id)
    modalities = (result or {}).get("input_modalities") or infer_modalities(metadata, settings.incoming_dir / take_id)
    modality_labels = _modality_labels(modalities, metadata if isinstance(metadata, dict) else {})
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
        modality_labels=modality_labels,
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
        acquisition_processing_status=summary.acquisition_processing_status,
    )


def _resolve_specific_run(settings: ApiSettings, take_id: str, run_id: str) -> tuple[dict[str, Any], Path] | None:
    from vision_3d_acquisition.pipelines.comparison import resolve_pipeline_run_source

    entries = process_entries_for_take(settings.data_dir, take_id)
    entry = next((item for item in entries if str(item.get("run_id") or "") == run_id), None)
    pipeline_id = str((entry or {}).get("pipeline_id") or "") or None
    if pipeline_id is None:
        return None
    resolved = resolve_pipeline_run_source(settings, pipeline_id=pipeline_id, take_id=take_id, run_id=run_id)
    if resolved is None:
        return None
    artifact_root = resolved.get("artifact_root")
    if not isinstance(artifact_root, Path):
        return None
    return dict(resolved.get("result_payload") or {}), artifact_root


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
        "run_id": run_id,
    }
    classification_artifact = next((item for item in artifacts if str(item.get("artifact_id") or "") in {"classification_result", "classification_result_artifact"}), None)
    classification_meta = (classification_artifact or {}).get("metadata") if isinstance(classification_artifact, dict) else None
    classification_entries = []
    if isinstance(classification_meta, dict) and isinstance(classification_meta.get("entries"), list):
        classification_entries = [entry for entry in classification_meta["entries"] if isinstance(entry, dict)]
    mapped_objects: list[dict[str, Any]] = []
    for entry in classification_entries:
        object_id = int(entry.get("object_id") or 0)
        geometry = entry.get("geometry") if isinstance(entry.get("geometry"), dict) else {}
        mapped_objects.append(
            {
                "object_id": object_id,
                "class_name": str(entry.get("class_label") or "unknown"),
                "subclass_label": entry.get("subclass_label"),
                "confidence": entry.get("confidence"),
                "point_count": None,
                "center_mm": None,
                "dimensions_mm": None,
                "diameter_estimate_mm": None,
                "diameter_mm": None,
                "sphericity_score": geometry.get("circularity"),
                "fit_rmse_mm": entry.get("fit_rmse"),
                "height_above_belt_mm": None,
                "fraction_points_inside_belt": None,
                "filter_status": "kept",
                "filter_reason": None,
                "artifact_ids": [],
                "source_candidate_id": entry.get("source_candidate_id"),
                "decision_reasons": entry.get("decision_reasons"),
                "geometry": geometry,
            }
        )
    result["objects"] = mapped_objects
    if isinstance(classification_meta, dict):
        result["classification_diagnostics"] = {
            "candidates_from_ellipse_fitting": int(classification_meta.get("candidates_from_ellipse_fitting") or 0),
            "candidates_classified": int(classification_meta.get("candidates_classified") or 0),
            "skipped_reason": classification_meta.get("skipped_reason"),
        }
    if isinstance(result.get("summary"), dict):
        result["summary"]["object_count"] = len(mapped_objects)
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
    result["object_candidates"] = normalize_object_candidates(result)
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
    # Resolve the filename relative to each indexed run's own output directory. This
    # covers run-scoped artifacts, including copied "reused/{stage_id}/..." files under
    # a partial-rerun child run's output directory.
    for root in allowed_roots:
        candidate = (root / normalized).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    shared_candidate = (settings.data_dir / normalized).resolve()
    try:
        shared_candidate.relative_to((settings.data_dir / "processes" / "runs").resolve())
    except ValueError:
        return None
    if not shared_candidate.is_file():
        return None
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
    files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
    height_preview = files.get("heightmap_preview")
    if isinstance(height_preview, str) and height_preview:
        return height_preview
    assets = assets_by_modality(metadata, settings.incoming_dir / take_id)
    for modality in ("reflectance", "rgb", "laser_rgb"):
        group = assets.get(modality)
        if isinstance(group, dict):
            for filename in group.values():
                if filename:
                    return str(filename)
    result_files = result.get("files") if isinstance(result.get("files"), dict) else {}
    raw_upload = files.get("raw_upload")
    if isinstance(raw_upload, str) and raw_upload:
        return raw_upload
    for key in ("height_segmentation_overlay", "segmentation_overlay", "overlay", "debug_segmentation", "debug_plane_segmentation"):
        candidate = result_files.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _take_run_history(settings: ApiSettings, take_id: str, result: dict[str, Any] | None) -> list[dict[str, Any]]:
    from vision_3d_acquisition.pipelines.run_lineage import enrich_run_entry

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
                "execution_mode": "full_run",
                "run_type": "legacy_3d_result",
                "parent_run_id": None,
            }
        )
    for entry in sorted(process_entries_for_take(settings.data_dir, take_id), key=lambda item: str(item.get("created_at") or ""), reverse=True):
        enriched = enrich_run_entry(entry)
        history.append(
            {
                "run_id": enriched.get("run_id"),
                "pipeline": enriched.get("pipeline_family"),
                "timestamp": enriched.get("created_at"),
                "status": enriched.get("status"),
                "duration_ms": None,
                "backend": "process_service",
                "pipeline_instance_id": enriched.get("pipeline_instance_id"),
                "execution_mode": enriched.get("execution_mode"),
                "run_type": enriched.get("run_type"),
                "parent_run_id": enriched.get("parent_run_id"),
                "boundary_stage_id": enriched.get("boundary_stage_id"),
                "comparison_id": enriched.get("comparison_id"),
                "summary": enriched.get("summary"),
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


def _modality_labels(modalities: list[str], metadata: dict[str, Any]) -> list[str]:
    source = metadata.get("source")
    source_payload = source if isinstance(source, dict) else {}
    labels: list[str] = []
    for modality in modalities:
        if modality == "heightmap" and str(source_payload.get("sensor") or "") == "trispector_2_5d":
            labels.append("heightmap_2_5d")
        else:
            labels.append(modality)
    return labels
