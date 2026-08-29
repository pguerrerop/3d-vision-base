from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from vision_3d_acquisition.api.filesystem import get_take_detail, list_takes_paged
from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.registry import list_pipelines

ProcessingJobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "partial"]
ProcessingJobScopeType = Literal["take_selection", "dataset", "dataset_session", "acquisition_session", "filter_query"]
ProcessingReprocessMode = Literal["full", "failed_only", "missing_outputs"]


class ProcessingJobTakeSelection(BaseModel):
    take_ids: list[str] = Field(default_factory=list)
    dataset_id: str | None = None
    session_id: str | None = None
    filters: dict[str, Any] | None = None


class ProcessingJobCreateRequest(BaseModel):
    take_selection: ProcessingJobTakeSelection | None = None
    scope_type: ProcessingJobScopeType | None = None
    pipeline_id: str
    dataset_id: str | None = None
    dataset_session_id: str | None = None
    acquisition_session_id: str | None = None
    take_ids: list[str] = Field(default_factory=list)
    filters_snapshot: dict[str, Any] | None = None
    reprocess_mode: ProcessingReprocessMode = "full"
    skip_completed: bool = False
    force: bool = False
    selected_stage: str | None = None
    classifier_rules_path: str | None = None
    created_by: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ProcessingJobCancelResponse(BaseModel):
    ok: bool
    job_id: str
    status: ProcessingJobStatus


class ProcessingJobItem(BaseModel):
    take_id: str
    status: Literal["queued", "running", "completed", "failed", "skipped", "cancelled"] = "queued"
    run_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    skipped_reason: str | None = None


class ProcessingJobRecord(BaseModel):
    id: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    status: ProcessingJobStatus = "queued"
    scope_type: ProcessingJobScopeType
    pipeline_id: str
    pipeline_family: str
    total_takes: int = 0
    completed_takes: int = 0
    failed_takes: int = 0
    skipped_takes: int = 0
    created_by: str | None = None
    filters_summary: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    error_summary: list[dict[str, Any]] = Field(default_factory=list)
    take_items: list[ProcessingJobItem] = Field(default_factory=list)
    cancelled_at: str | None = None


@dataclass(frozen=True)
class ProcessingJobPaths:
    root: Path

    @property
    def job_dir(self) -> Path:
        return self.root / "runtime" / "processing_jobs"

    def job_path(self, job_id: str) -> Path:
        return self.job_dir / f"{job_id}.json"

    def progress_path(self, job_id: str) -> Path:
        return self.job_dir / f"{job_id}.jsonl"


class ProcessingJobService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.paths = ProcessingJobPaths(settings.data_dir)
        self.paths.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_threads: dict[str, threading.Thread] = {}

    def create_job(self, payload: ProcessingJobCreateRequest) -> ProcessingJobRecord:
        take_ids, scope_type, filters_summary = self._resolve_take_ids(payload)
        pipeline = self._resolve_pipeline(payload.pipeline_id)
        job_id = f"pj_{uuid.uuid4().hex[:12]}"
        now = _now()
        items = [ProcessingJobItem(take_id=take_id) for take_id in take_ids]
        summary = _build_scope_summary(scope_type, take_ids=take_ids, filters_summary=filters_summary)
        record = ProcessingJobRecord(
            id=job_id,
            created_at=now,
            status="queued",
            scope_type=scope_type,
            pipeline_id=payload.pipeline_id,
            pipeline_family=str(pipeline.get("pipeline_family") or "generic"),
            total_takes=len(take_ids),
            created_by=payload.created_by,
            filters_summary=filters_summary,
            options={
                "reprocess_mode": payload.reprocess_mode,
                "skip_completed": bool(payload.skip_completed),
                "force": bool(payload.force),
                "selected_stage": payload.selected_stage,
                "classifier_rules_path": payload.classifier_rules_path,
                **payload.options,
            },
            summary=summary,
            take_items=items,
        )
        self._write_job(record)
        self._append_progress(job_id, {"event": "created", "created_at": now, "take_count": len(take_ids)})
        self._start_worker(job_id)
        return self.load_job(job_id) or record

    def list_jobs(self) -> list[ProcessingJobRecord]:
        jobs: list[ProcessingJobRecord] = []
        for path in sorted(self.paths.job_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            record = self._read_job(path)
            if record is not None:
                jobs.append(record)
        return jobs

    def load_job(self, job_id: str) -> ProcessingJobRecord | None:
        return self._read_job(self.paths.job_path(job_id))

    def cancel_job(self, job_id: str) -> ProcessingJobRecord:
        record = self.load_job(job_id)
        if record is None:
            raise FileNotFoundError(f"Unknown processing job: {job_id}")
        if record.status in {"completed", "failed", "cancelled", "partial"}:
            return record
        event = self._cancel_events.setdefault(job_id, threading.Event())
        event.set()
        updated = record.model_copy(update={
            "status": "cancelled",
            "cancelled_at": _now(),
            "completed_at": record.completed_at or _now(),
            "summary": {**record.summary, "cancel_requested": True},
        })
        self._write_job(updated)
        self._append_progress(job_id, {"event": "cancel_requested", "timestamp": _now()})
        return updated

    def _start_worker(self, job_id: str) -> None:
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        self._active_threads[job_id] = thread
        thread.start()

    def _run_job(self, job_id: str) -> None:
        record = self.load_job(job_id)
        if record is None:
            return
        event = self._cancel_events.setdefault(job_id, threading.Event())
        record = record.model_copy(update={"status": "running", "started_at": record.started_at or _now()})
        self._write_job(record)
        self._append_progress(job_id, {"event": "started", "timestamp": record.started_at})
        completed = 0
        failed = 0
        skipped = 0
        errors: list[dict[str, Any]] = []
        items: list[ProcessingJobItem] = []
        try:
            for raw_item in record.take_items:
                current = raw_item.model_copy(update={"status": "running", "started_at": raw_item.started_at or _now()})
                self._update_job_item(job_id, current)
                if event.is_set():
                    skipped += 1
                    cancelled_item = current.model_copy(update={
                        "status": "cancelled",
                        "completed_at": _now(),
                        "skipped_reason": "cancelled",
                    })
                    items.append(cancelled_item)
                    self._update_job_item(job_id, cancelled_item)
                    continue
                try:
                    action = self._dispatch_take(record, current.take_id)
                    if bool(action.get("skipped")):
                        skipped += 1
                        skipped_item = current.model_copy(update={
                            "status": "skipped",
                            "completed_at": _now(),
                            "skipped_reason": str(action.get("skipped_reason") or "filtered_by_job_mode"),
                        })
                        self._update_job_item(job_id, skipped_item)
                        items.append(skipped_item)
                        continue
                    status = str(action.get("status") or action.get("result", {}).get("status") or "completed")
                    run_id = str(action.get("run_id") or action.get("result", {}).get("run_id") or "") or None
                    completed_item = current.model_copy(update={
                        "status": "completed" if status not in {"failed", "error"} else "failed",
                        "completed_at": _now(),
                        "run_id": run_id,
                    })
                    self._update_job_item(job_id, completed_item)
                    items.append(completed_item)
                    if completed_item.status == "completed":
                        completed += 1
                    else:
                        failed += 1
                        errors.append({"take_id": current.take_id, "error": str(action)})
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    error_item = current.model_copy(update={
                        "status": "failed",
                        "completed_at": _now(),
                        "error": str(exc),
                    })
                    self._update_job_item(job_id, error_item)
                    items.append(error_item)
                    errors.append({"take_id": current.take_id, "error": str(exc)})
            final_status: ProcessingJobStatus
            if event.is_set() and failed == 0 and completed == 0:
                final_status = "cancelled"
            elif failed and completed:
                final_status = "partial"
            elif failed and not completed:
                final_status = "failed"
            elif event.is_set() and completed:
                final_status = "partial"
            else:
                final_status = "completed"
            updated = record.model_copy(update={
                "status": final_status,
                "completed_at": _now(),
                "completed_takes": completed,
                "failed_takes": failed,
                "skipped_takes": skipped,
                "take_items": items,
                "summary": {
                    **record.summary,
                    "completed": completed,
                    "failed": failed,
                    "skipped": skipped,
                    "total": record.total_takes,
                },
                "error_summary": errors,
            })
            self._write_job(updated)
            self._append_progress(job_id, {"event": "finished", "timestamp": updated.completed_at, "status": updated.status, "completed": completed, "failed": failed, "skipped": skipped})
        finally:
            self._active_threads.pop(job_id, None)

    def _dispatch_take(self, job: ProcessingJobRecord, take_id: str) -> dict[str, Any]:
        details = get_take_detail(self.settings, take_id)
        if details is None:
            raise FileNotFoundError(f"Unknown take: {take_id}")
        if not _should_reprocess_take(
            self.settings,
            take_id=take_id,
            pipeline_family=job.pipeline_family,
            reprocess_mode=str(job.options.get("reprocess_mode") or "full"),
            skip_completed=bool(job.options.get("skip_completed")),
            force=bool(job.options.get("force")),
        ):
            self._append_progress(job.id, {"event": "skipped", "take_id": take_id, "reason": "filtered_by_job_mode", "timestamp": _now()})
            return {"ok": True, "take_id": take_id, "skipped": True, "skipped_reason": "filtered_by_job_mode", "status": "skipped"}
        stage_params = None
        classifier_rules_path = str(job.options.get("classifier_rules_path") or "") or None
        if classifier_rules_path:
            stage_params = {"classify_25d": {"classifier_rules_path": classifier_rules_path}}
        return dispatch_take_processing(
            settings=self.settings,
            take_id=take_id,
            pipeline_id=job.pipeline_id,
            reprocess=True,
            source_id=None,
            recipe_version_id=None,
            acquisition_group_id=None,
            calibration_profile_id=None,
            stage_params=stage_params,
        )

    def _resolve_take_ids(self, payload: ProcessingJobCreateRequest) -> tuple[list[str], ProcessingJobScopeType, dict[str, Any]]:
        take_selection = payload.take_selection
        explicit_take_ids = [str(item).strip() for item in (take_selection.take_ids if take_selection else payload.take_ids) if str(item).strip()]
        if explicit_take_ids:
            deduped = list(dict.fromkeys(explicit_take_ids))
            return deduped, "take_selection", {"take_ids": deduped}

        dataset_id = payload.dataset_id or (take_selection.dataset_id if take_selection else None)
        dataset_session_id = payload.dataset_session_id or (take_selection.session_id if take_selection else None)
        acquisition_session_id = payload.acquisition_session_id or None
        filters_raw = payload.filters_snapshot or (take_selection.filters if take_selection else None) or {}
        filters = {str(key): value for key, value in filters_raw.items() if value is not None}
        if dataset_id:
            filters["dataset_id"] = dataset_id
        include_archived = bool(filters.pop("show_archived", False))
        if dataset_session_id:
            scope_type: ProcessingJobScopeType = "dataset_session"
        elif acquisition_session_id:
            scope_type = "acquisition_session"
        elif dataset_id:
            scope_type = "dataset"
        else:
            scope_type = "filter_query"
        resolved = resolve_processing_job_take_ids(
            self.settings,
            dataset_id=dataset_id,
            dataset_session_id=dataset_session_id,
            acquisition_session_id=acquisition_session_id,
            filters=filters,
            include_archived=include_archived,
        )
        filters_summary = dict(filters)
        if dataset_session_id:
            filters_summary["dataset_session_id"] = dataset_session_id
        if acquisition_session_id:
            filters_summary["acquisition_session_id"] = acquisition_session_id
        return resolved, scope_type, filters_summary

    def _resolve_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        pipeline = next((item for item in list_pipelines() if item.get("id") == pipeline_id), None)
        if pipeline is None:
            raise ValueError(f"Unknown pipeline: {pipeline_id}")
        return pipeline

    def _read_job(self, path: Path) -> ProcessingJobRecord | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        take_items = [
            ProcessingJobItem.model_validate(item)
            for item in payload.get("take_items", [])
            if isinstance(item, dict)
        ]
        payload["take_items"] = take_items
        try:
            return ProcessingJobRecord.model_validate(payload)
        except Exception:
            return None

    def _write_job(self, record: ProcessingJobRecord) -> None:
        _write_json(self.paths.job_path(record.id), record.model_dump(mode="json"))

    def _update_job_item(self, job_id: str, item: ProcessingJobItem) -> None:
        current = self.load_job(job_id)
        if current is None:
            return
        updated_items = [item if existing.take_id == item.take_id else existing for existing in current.take_items]
        if not any(existing.take_id == item.take_id for existing in current.take_items):
            updated_items.append(item)
        summary = _summarize_items(updated_items)
        record = current.model_copy(update={
            "take_items": updated_items,
            "completed_takes": summary["completed"],
            "failed_takes": summary["failed"],
            "skipped_takes": summary["skipped"],
            "summary": {**current.summary, **summary},
        })
        self._write_job(record)

    def _append_progress(self, job_id: str, payload: dict[str, Any]) -> None:
        path = self.paths.progress_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {**payload, "job_id": job_id}
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, sort_keys=True))
                handle.write("\n")


_SERVICES: dict[str, ProcessingJobService] = {}
_SERVICES_LOCK = threading.Lock()


def get_processing_job_service(settings: ApiSettings) -> ProcessingJobService:
    key = str(settings.data_dir.resolve())
    with _SERVICES_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = ProcessingJobService(settings)
            _SERVICES[key] = service
        return service


def resolve_processing_job_take_ids(
    settings: ApiSettings,
    *,
    dataset_id: str | None = None,
    dataset_session_id: str | None = None,
    acquisition_session_id: str | None = None,
    filters: dict[str, Any] | None = None,
    explicit_take_ids: list[str] | None = None,
    include_archived: bool = False,
) -> list[str]:
    if explicit_take_ids:
        return [str(item).strip() for item in explicit_take_ids if str(item).strip()]
    resolved_filters = dict(filters or {})
    if dataset_id:
        resolved_filters["dataset_id"] = dataset_id
    if dataset_session_id:
        resolved_filters["dataset_session_id"] = dataset_session_id
    if acquisition_session_id:
        resolved_filters["acquisition_session_id"] = acquisition_session_id
    resolved_filters.pop("show_archived", None)
    page_offset = 0
    take_ids: list[str] = []
    while True:
        page = list_takes_paged(
            settings,
            limit=200,
            offset=page_offset,
            dataset_id=str(resolved_filters.get("dataset_id") or "") or None,
            validation_status=str(resolved_filters.get("validation_status") or "") or None,
            tag=str(resolved_filters.get("tag") or "") or None,
            search=str(resolved_filters.get("search") or "") or None,
            created_from=str(resolved_filters.get("created_from") or "") or None,
            created_to=str(resolved_filters.get("created_to") or "") or None,
            expected_class=str(resolved_filters.get("expected_class") or "") or None,
            split=str(resolved_filters.get("split") or "") or None,
            calibration_linkage_only=bool(resolved_filters.get("calibration_linkage_only") or False),
            physical_object_id=str(resolved_filters.get("physical_object_id") or "") or None,
            session_type=str(resolved_filters.get("session_type") or "") or None,
            category=str(resolved_filters.get("category") or "") or None,
            reference_type=str(resolved_filters.get("reference_type") or "") or None,
            is_reference=resolved_filters.get("is_reference"),
            is_golden_sample=resolved_filters.get("is_golden_sample"),
            include_archived=include_archived,
        )
        items = page.get("items") or []
        page_take_ids = [str(item.take_id) for item in items if getattr(item, "take_id", None)]
        if dataset_session_id or acquisition_session_id:
            page_take_ids = [
                take_id
                for take_id in page_take_ids
                if _take_matches_session_scope(settings, take_id, dataset_session_id=dataset_session_id, acquisition_session_id=acquisition_session_id)
            ]
        take_ids.extend(page_take_ids)
        if not page.get("has_more"):
            break
        page_offset = int(page.get("next_offset") or (page_offset + len(items)))
    # `list_takes_paged` orders newest-first for browsing UX; job execution
    # wants a stable, reproducible order independent of that, so sort here.
    return sorted(dict.fromkeys(take_ids))


def _summarize_items(items: list[ProcessingJobItem]) -> dict[str, int]:
    summary = {"completed": 0, "failed": 0, "skipped": 0}
    for item in items:
        if item.status == "completed":
            summary["completed"] += 1
        elif item.status == "failed":
            summary["failed"] += 1
        elif item.status in {"skipped", "cancelled"}:
            summary["skipped"] += 1
    return summary


def _build_scope_summary(
    scope_type: ProcessingJobScopeType,
    *,
    take_ids: list[str],
    filters_summary: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "take_ids": take_ids,
        "scope_type": scope_type,
        "total": len(take_ids),
    }
    dataset_id = str(filters_summary.get("dataset_id") or "") or None
    dataset_session_id = str(filters_summary.get("dataset_session_id") or filters_summary.get("session_id") or "") or None
    acquisition_session_id = str(filters_summary.get("acquisition_session_id") or "") or None
    if dataset_id:
        summary["dataset_id"] = dataset_id
    if dataset_session_id:
        summary["dataset_session_id"] = dataset_session_id
    if acquisition_session_id:
        summary["acquisition_session_id"] = acquisition_session_id
    if scope_type == "dataset_session":
        summary["scope_label"] = f"Experiment session {dataset_session_id or 'selected'}"
    elif scope_type == "acquisition_session":
        summary["scope_label"] = f"Runtime acquisition group {acquisition_session_id or 'selected'}"
    elif scope_type == "dataset":
        summary["scope_label"] = f"Dataset {dataset_id or 'selected'}"
    elif scope_type == "take_selection":
        summary["scope_label"] = "Selected takes"
    else:
        summary["scope_label"] = "Filtered takes"
    return summary


def _should_reprocess_take(
    settings: ApiSettings,
    *,
    take_id: str,
    pipeline_family: str,
    reprocess_mode: str,
    skip_completed: bool,
    force: bool,
) -> bool:
    if force:
        return True
    detail = get_take_detail(settings, take_id)
    if detail is None:
        return False
    processing_summary = next((item for item in (detail.processing_by_family or []) if str(item.get("family") or "") == pipeline_family), None)
    has_completed_output = bool((processing_summary or {}).get("hasCompletedOutput"))
    status = str((processing_summary or {}).get("status") or "").lower()
    if skip_completed and has_completed_output:
        return False
    if reprocess_mode == "missing_outputs":
        return not has_completed_output
    if reprocess_mode == "failed_only":
        return status in {"failed", "partial"}
    return True


def _take_matches_session_scope(
    settings: ApiSettings,
    take_id: str,
    *,
    dataset_session_id: str | None,
    acquisition_session_id: str | None,
) -> bool:
    if dataset_session_id:
        detail = get_take_detail(settings, take_id)
        if detail is None:
            return False
        if str((detail.take_metadata or {}).get("session_id") or "") != dataset_session_id:
            return False
    if acquisition_session_id:
        summary = get_take_detail(settings, take_id)
        if summary is None:
            return False
        if str(summary.session_id or "") != acquisition_session_id:
            return False
    return True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
