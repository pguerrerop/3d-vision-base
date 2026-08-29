from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from vision_3d_acquisition.api.feature_catalog import (
    feature_definition_for_key,
    feature_job_backfill_keys,
    feature_job_reprocess_keys,
)
from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.ml.features.surface_sphere_fit_workflow import (
    PIPELINE_ID_25D,
    backfill_take_feature,
    inspect_take_feature,
    take_needs_feature_action,
)
from vision_3d_acquisition.pipelines.registry import list_pipelines

FeatureJobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "partial"]
FeatureJobScope = Literal["ml_set", "dataset", "takes"]
FeatureJobMode = Literal["reprocess", "backfill"]


class FeatureJobRequest(BaseModel):
    scope: FeatureJobScope
    dataset_id: str | None = None
    ml_set_id: str | None = None
    take_ids: list[str] = Field(default_factory=list)
    feature_key: str
    mode: FeatureJobMode
    pipeline_id: str | None = None
    only_missing: bool = False


class FeatureJobItem(BaseModel):
    take_id: str
    status: Literal["queued", "running", "completed", "failed", "skipped", "cancelled"] = "queued"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    skipped_reason: str | None = None
    values_written: int = 0


class FeatureJobRecord(BaseModel):
    id: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    status: FeatureJobStatus = "queued"
    scope: FeatureJobScope
    dataset_id: str | None = None
    ml_set_id: str | None = None
    feature_key: str
    feature_display_name: str
    mode: FeatureJobMode
    pipeline_id: str | None = None
    only_missing: bool = False
    total_takes: int = 0
    completed_takes: int = 0
    failed_takes: int = 0
    skipped_takes: int = 0
    values_written: int = 0
    current_take_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    error_summary: list[dict[str, Any]] = Field(default_factory=list)
    take_items: list[FeatureJobItem] = Field(default_factory=list)
    cancelled_at: str | None = None


@dataclass(frozen=True)
class FeatureJobPaths:
    root: Path

    @property
    def job_dir(self) -> Path:
        return self.root / "runtime" / "feature_jobs"

    def job_path(self, job_id: str) -> Path:
        return self.job_dir / f"{job_id}.json"

    def progress_path(self, job_id: str) -> Path:
        return self.job_dir / f"{job_id}.jsonl"


class FeatureJobService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.paths = FeatureJobPaths(settings.data_dir)
        self.paths.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_threads: dict[str, threading.Thread] = {}

    def create_job(self, payload: FeatureJobRequest) -> FeatureJobRecord:
        self._validate_request(payload)
        take_ids = self._resolve_take_ids(payload)
        if payload.only_missing:
            take_ids = [
                take_id
                for take_id in take_ids
                if take_needs_feature_action(
                    self.settings.processed_dir,
                    take_id,
                    payload.feature_key,
                    payload.mode,
                    only_missing=True,
                )
            ]
        if not take_ids:
            raise ValueError("no takes matched the selected scope and only_missing filter")

        definition = feature_definition_for_key(payload.feature_key)
        pipeline_id = payload.pipeline_id or PIPELINE_ID_25D
        if payload.mode == "reprocess":
            self._resolve_pipeline(pipeline_id)

        job_id = f"fj_{uuid.uuid4().hex[:12]}"
        now = _now()
        items = [FeatureJobItem(take_id=take_id) for take_id in take_ids]
        record = FeatureJobRecord(
            id=job_id,
            created_at=now,
            status="queued",
            scope=payload.scope,
            dataset_id=payload.dataset_id,
            ml_set_id=payload.ml_set_id,
            feature_key=payload.feature_key,
            feature_display_name=definition.display_name,
            mode=payload.mode,
            pipeline_id=pipeline_id if payload.mode == "reprocess" else None,
            only_missing=bool(payload.only_missing),
            total_takes=len(take_ids),
            summary={
                "scope": payload.scope,
                "dataset_id": payload.dataset_id,
                "ml_set_id": payload.ml_set_id,
                "feature_key": payload.feature_key,
                "mode": payload.mode,
                "only_missing": bool(payload.only_missing),
            },
            take_items=items,
        )
        self._write_job(record)
        self._append_progress(job_id, {"event": "created", "created_at": now, "take_count": len(take_ids)})
        self._start_worker(job_id)
        return self.load_job(job_id) or record

    def load_job(self, job_id: str) -> FeatureJobRecord | None:
        return self._read_job(self.paths.job_path(job_id))

    def cancel_job(self, job_id: str) -> FeatureJobRecord:
        record = self.load_job(job_id)
        if record is None:
            raise FileNotFoundError(f"Unknown feature job: {job_id}")
        if record.status in {"completed", "failed", "cancelled", "partial"}:
            return record
        event = self._cancel_events.setdefault(job_id, threading.Event())
        event.set()
        updated = record.model_copy(
            update={
                "status": "cancelled",
                "cancelled_at": _now(),
                "completed_at": record.completed_at or _now(),
                "summary": {**record.summary, "cancel_requested": True},
            }
        )
        self._write_job(updated)
        self._append_progress(job_id, {"event": "cancel_requested", "timestamp": _now()})
        return updated

    def _validate_request(self, payload: FeatureJobRequest) -> None:
        if payload.mode == "reprocess":
            supported = feature_job_reprocess_keys()
            if payload.feature_key not in supported:
                supported_list = ", ".join(sorted(supported))
                raise ValueError(
                    f"unsupported feature_key for reprocess: {payload.feature_key}. "
                    f"Supported sphere-consistency keys: {supported_list}"
                )
        elif payload.feature_key not in feature_job_backfill_keys():
            supported = ", ".join(sorted(feature_job_backfill_keys()))
            raise ValueError(
                f"unsupported feature_key for backfill: {payload.feature_key}. Supported: {supported}"
            )
        if payload.scope == "ml_set" and not str(payload.ml_set_id or "").strip():
            raise ValueError("ml_set_id is required when scope=ml_set")
        if payload.scope == "dataset" and not str(payload.dataset_id or "").strip():
            raise ValueError("dataset_id is required when scope=dataset")
        if payload.scope == "takes" and not payload.take_ids:
            raise ValueError("take_ids is required when scope=takes")
        if payload.mode == "reprocess" and not str(payload.pipeline_id or PIPELINE_ID_25D).strip():
            raise ValueError("pipeline_id is required for reprocess mode")

    def _resolve_take_ids(self, payload: FeatureJobRequest) -> list[str]:
        from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

        if payload.scope == "takes":
            return list(dict.fromkeys(str(item).strip() for item in payload.take_ids if str(item).strip()))

        service = DatasetService(self.settings.data_dir)
        if payload.scope == "ml_set":
            ml_set = service.resolve_ml_set(ml_set_id=str(payload.ml_set_id), dataset_id=payload.dataset_id)
            resolved_dataset_id = str(ml_set.get("dataset_id") or "")
            memberships = service.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=str(payload.ml_set_id))
            take_ids = sorted(
                {
                    str(row.get("take_id") or "")
                    for row in memberships
                    if row.get("include", True) and str(row.get("take_id") or "")
                }
            )
            if not take_ids:
                raise ValueError(f"no included takes found for ML set: {payload.ml_set_id}")
            return take_ids

        dataset_id = str(payload.dataset_id or "")
        take_ids = sorted(
            {
                take_id
                for _session_id, take_id, _payload in service.iter_dataset_takes(dataset_id)
                if take_id
            }
        )
        if not take_ids:
            raise ValueError(f"no takes found for dataset: {dataset_id}")
        return take_ids

    def _resolve_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        pipeline = next((item for item in list_pipelines() if item.get("id") == pipeline_id), None)
        if pipeline is None:
            raise ValueError(f"unknown pipeline: {pipeline_id}")
        return pipeline

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
        values_written = 0
        errors: list[dict[str, Any]] = []
        items: list[FeatureJobItem] = []

        try:
            for raw_item in record.take_items:
                if event.is_set():
                    skipped += 1
                    cancelled_item = raw_item.model_copy(
                        update={
                            "status": "cancelled",
                            "completed_at": _now(),
                            "skipped_reason": "cancelled",
                        }
                    )
                    items.append(cancelled_item)
                    self._update_job_item(
                        job_id,
                        cancelled_item,
                        current_take_id=None,
                        completed_takes=completed,
                        failed_takes=failed,
                        skipped_takes=skipped,
                        values_written=values_written,
                    )
                    continue

                current = raw_item.model_copy(update={"status": "running", "started_at": raw_item.started_at or _now()})
                self._update_job_item(job_id, current, current_take_id=current.take_id)

                try:
                    if record.mode == "reprocess":
                        action = dispatch_take_processing(
                            settings=self.settings,
                            take_id=current.take_id,
                            pipeline_id=str(record.pipeline_id or PIPELINE_ID_25D),
                            reprocess=True,
                            source_id=None,
                            recipe_version_id=None,
                            acquisition_group_id=None,
                            calibration_profile_id=None,
                            stage_params=None,
                        )
                        status = str(action.get("status") or action.get("result", {}).get("status") or "completed")
                        item_status = "completed" if status not in {"failed", "error"} else "failed"
                        finished = current.model_copy(update={"status": item_status, "completed_at": _now()})
                        if item_status == "completed":
                            completed += 1
                        else:
                            failed += 1
                            errors.append({"take_id": current.take_id, "error": str(action)})
                    else:
                        result = backfill_take_feature(
                            processed_dir=self.settings.processed_dir,
                            take_id=current.take_id,
                            incoming_dir=self.settings.incoming_dir,
                            feature_key=record.feature_key,
                            dry_run=False,
                        )
                        written = int(result.get("objects_with_value") or 0)
                        if written == 0 and int(result.get("entries_written") or 0) == 0:
                            skipped += 1
                            finished = current.model_copy(
                                update={
                                    "status": "skipped",
                                    "completed_at": _now(),
                                    "skipped_reason": "already_has_canonical_value",
                                    "values_written": 0,
                                }
                            )
                        else:
                            completed += 1
                            values_written += written
                            finished = current.model_copy(
                                update={
                                    "status": "completed",
                                    "completed_at": _now(),
                                    "values_written": written,
                                }
                            )
                    items.append(finished)
                    self._update_job_item(
                        job_id,
                        finished,
                        current_take_id=None,
                        completed_takes=completed,
                        failed_takes=failed,
                        skipped_takes=skipped,
                        values_written=values_written,
                    )
                    self._append_progress(
                        job_id,
                        {
                            "event": "take_finished",
                            "take_id": current.take_id,
                            "status": finished.status,
                            "values_written": finished.values_written,
                            "timestamp": _now(),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    error_item = current.model_copy(update={"status": "failed", "completed_at": _now(), "error": str(exc)})
                    items.append(error_item)
                    errors.append({"take_id": current.take_id, "error": str(exc)})
                    self._update_job_item(
                        job_id,
                        error_item,
                        current_take_id=None,
                        completed_takes=completed,
                        failed_takes=failed,
                        skipped_takes=skipped,
                        values_written=values_written,
                    )

            final_status: FeatureJobStatus
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

            updated = record.model_copy(
                update={
                    "status": final_status,
                    "completed_at": _now(),
                    "completed_takes": completed,
                    "failed_takes": failed,
                    "skipped_takes": skipped,
                    "values_written": values_written,
                    "current_take_id": None,
                    "take_items": items,
                    "summary": {
                        **record.summary,
                        "completed": completed,
                        "failed": failed,
                        "skipped": skipped,
                        "values_written": values_written,
                        "total": record.total_takes,
                    },
                    "error_summary": errors,
                }
            )
            self._write_job(updated)
            self._append_progress(
                job_id,
                {
                    "event": "finished",
                    "timestamp": updated.completed_at,
                    "status": updated.status,
                    "completed": completed,
                    "failed": failed,
                    "skipped": skipped,
                    "values_written": values_written,
                },
            )
        finally:
            self._active_threads.pop(job_id, None)

    def _update_job_item(
        self,
        job_id: str,
        item: FeatureJobItem,
        *,
        current_take_id: str | None,
        completed_takes: int | None = None,
        failed_takes: int | None = None,
        skipped_takes: int | None = None,
        values_written: int | None = None,
    ) -> None:
        with self._lock:
            record = self.load_job(job_id)
            if record is None:
                return
            items = list(record.take_items)
            for idx, existing in enumerate(items):
                if existing.take_id == item.take_id:
                    items[idx] = item
                    break
            updates: dict[str, Any] = {"take_items": items, "current_take_id": current_take_id}
            if completed_takes is not None:
                updates["completed_takes"] = completed_takes
            if failed_takes is not None:
                updates["failed_takes"] = failed_takes
            if skipped_takes is not None:
                updates["skipped_takes"] = skipped_takes
            if values_written is not None:
                updates["values_written"] = values_written
            if completed_takes is not None or failed_takes is not None or skipped_takes is not None:
                updates["summary"] = {
                    **record.summary,
                    "completed": completed_takes if completed_takes is not None else record.completed_takes,
                    "failed": failed_takes if failed_takes is not None else record.failed_takes,
                    "skipped": skipped_takes if skipped_takes is not None else record.skipped_takes,
                    "values_written": values_written if values_written is not None else record.values_written,
                    "total": record.total_takes,
                }
            updated = record.model_copy(update=updates)
            self._write_job(updated)

    def _write_job(self, record: FeatureJobRecord) -> None:
        self.paths.job_path(record.id).write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _read_job(self, path: Path) -> FeatureJobRecord | None:
        if not path.is_file():
            return None
        try:
            return FeatureJobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _append_progress(self, job_id: str, payload: dict[str, Any]) -> None:
        with self.paths.progress_path(job_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


_feature_job_services: dict[str, FeatureJobService] = {}


def get_feature_job_service(settings: ApiSettings) -> FeatureJobService:
    key = str(settings.data_dir)
    service = _feature_job_services.get(key)
    if service is None:
        service = FeatureJobService(settings)
        _feature_job_services[key] = service
    return service
