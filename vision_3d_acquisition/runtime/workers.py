from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.acquisition.process_integration import process_acquired_take_if_bound
from vision_3d_acquisition.acquisition.processing_contracts import AcquiredTakeReady
from vision_3d_acquisition.acquisition.processing_status import read_status
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.fusion.published_service import PublishedInspectionResultService
from vision_3d_acquisition.fusion.service import FusionService
from vision_3d_acquisition.processes.bindings import ProcessBindingService
from vision_3d_acquisition.runtime.processing_queue import TakeProcessingQueue
from vision_3d_acquisition.runtime.publication import PublishedInspectionResultServiceLite
from vision_3d_acquisition.runtime.worker_manager import RuntimeWorkerManager
from vision_3d_acquisition.runtime.worker_models import WorkerEvent, WorkerRunSummary, worker_now_iso

OK_STATUSES = {"ok", "success", "warning", "completed"}


class RuntimeWorkerBase:
    def __init__(
        self,
        *,
        settings: ApiSettings,
        manager: RuntimeWorkerManager,
        worker_id: str,
        worker_type: str,
        source_id: str | None = None,
        station_id: str | None = None,
        poll_interval_sec: float = 1.0,
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.source_id = source_id
        self.station_id = station_id
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self.queue = TakeProcessingQueue(settings.data_dir)
        self.manager.register_worker(
            worker_id=worker_id,
            worker_type=worker_type,
            source_id=source_id,
            station_id=station_id,
            poll_interval_sec=self.poll_interval_sec,
            enabled=True,
            config={"pipeline_id": None, "modality": None},
        )

    def run(self, *, once: bool, dry_run: bool) -> dict[str, Any]:
        summary = WorkerRunSummary(worker_id=self.worker_id, started_at=worker_now_iso())
        self.manager.clear_stop_request(self.worker_id)
        self._emit("WORKER_STARTED", f"{self.worker_type} started.", metadata={"once": once, "dry_run": dry_run})
        self.manager.heartbeat(self.worker_id, "running", {"worker_type": self.worker_type})
        while True:
            if self.manager.is_stop_requested(self.worker_id):
                self._emit("WORKER_STOP_REQUESTED", "Stop request observed by worker loop.")
                break
            result: dict[str, Any] = {}
            try:
                result = self.run_iteration(dry_run=dry_run)
                summary.processed_count += int(result.get("processed_count") or 0)
                summary.skipped_count += int(result.get("skipped_count") or 0)
                summary.error_count += int(result.get("error_count") or 0)
                summary.retry_count += int(result.get("retry_count") or 0)
            except Exception as exc:  # noqa: BLE001
                summary.error_count += 1
                summary.notes.append(str(exc))
                self.manager.mark_worker_error(self.worker_id, str(exc))
            self.manager.heartbeat(
                self.worker_id,
                "running",
                {
                    "processed_count": summary.processed_count,
                    "skipped_count": summary.skipped_count,
                    "error_count": summary.error_count,
                    "retry_count": summary.retry_count,
                    "activity": bool(result.get("activity")),
                    "success": bool(result.get("processed_count")),
                    "queue_depth": int(result.get("queue_depth") or 0),
                },
            )
            if once:
                break
            time.sleep(self.poll_interval_sec)
        summary.finished_at = worker_now_iso()
        final_state = "error" if summary.error_count > 0 else "stopped"
        self.manager.heartbeat(self.worker_id, final_state, summary.to_dict())
        self._emit("WORKER_FINISHED", f"{self.worker_type} finished.", metadata=summary.to_dict())
        return summary.to_dict()

    def run_iteration(self, *, dry_run: bool) -> dict[str, Any]:
        raise NotImplementedError

    def _emit(self, event_type: str, message: str, *, severity: str = "info", metadata: dict[str, Any] | None = None) -> None:
        self.manager.append_worker_event(
            self.worker_id,
            WorkerEvent(
                worker_id=self.worker_id,
                timestamp=worker_now_iso(),
                severity=severity,  # type: ignore[arg-type]
                event_type=event_type,
                message=message,
                metadata=dict(metadata or {}),
            ),
        )

    def _has_done_marker(self, take_id: str) -> bool:
        return _has_worker_done_marker(self.settings.data_dir, self.worker_id, take_id)

    def _touch_done_marker(self, take_id: str) -> None:
        _touch_worker_done_marker(self.settings.data_dir, self.worker_id, take_id)


class RgbAcquisitionProcessingWorker(RuntimeWorkerBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(worker_type="rgb_acquisition_processing_worker", **kwargs)
        self.manager.register_worker(
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            source_id=self.source_id,
            station_id=self.station_id,
            poll_interval_sec=self.poll_interval_sec,
            enabled=True,
            config={"pipeline_id": "mining_steel_ball_classification_2d", "modality": "rgb"},
        )

    def run_iteration(self, *, dry_run: bool) -> dict[str, Any]:
        processed = 0
        skipped = 0
        errors = 0
        retries = 0
        activity = False
        stale = self.queue.recover_stale_claims(claim_timeout_sec=120.0)
        if stale:
            self._emit("STALE_CLAIMS_RECOVERED", f"Recovered stale claims: {len(stale)}", metadata={"take_ids": stale})
            activity = True

        for take_id, metadata in _iter_incoming_take_metadata(self.settings.data_dir):
            modalities = [str(item) for item in (metadata.get("modalities") or [])]
            if "rgb" not in modalities:
                continue
            if _already_processed_take(self.settings.data_dir, take_id, pipeline_family="2d") or self._has_done_marker(take_id):
                continue
            acquisition_group_id = _ensure_take_group(self.settings, take_id, metadata, station_id=self.station_id)
            source_id = self.source_id or str(metadata.get("source") or "") or "usb_camera_0"
            queued = self.queue.enqueue(
                {
                    "take_id": take_id,
                    "source_id": source_id,
                    "modality": "rgb",
                    "modality_family": "2d",
                    "purpose": "acquisition_inspection",
                    "acquisition_group_id": acquisition_group_id,
                    "frameset_id": metadata.get("frameset", {}).get("frameset_id") if isinstance(metadata.get("frameset"), dict) else None,
                    "capture_timestamp": metadata.get("created_at"),
                    "metadata": {"worker_id": self.worker_id},
                    "max_retries": 3,
                }
            )
            if queued:
                activity = True

        while True:
            claim = self.queue.claim_next(worker_id=self.worker_id, modality="rgb", purpose="acquisition_inspection", claim_timeout_sec=120.0)
            if claim is None:
                break
            take_id = str(claim.get("take_id") or "")
            activity = True
            if dry_run:
                self.queue.release_claim(take_id)
                skipped += 1
                self._emit("RGB_WOULD_PROCESS", f"Dry-run: would process {take_id}", metadata={"take_id": take_id})
                continue
            decision = process_acquired_take_if_bound(
                self.settings,
                AcquiredTakeReady(
                    take_id=take_id,
                    source_id=str(claim.get("source_id") or self.source_id or "usb_camera_0"),
                    modality="rgb",
                    modality_family="2d",
                    acquisition_group_id=str(claim.get("acquisition_group_id") or "") or None,
                    metadata={"worker_id": self.worker_id},
                ),
                purpose="acquisition_inspection",
                auto_process=True,
            )
            if decision.ok and decision.status in {"processing_completed", "processing_binding_missing"}:
                processed += 1
                self.queue.mark_completed(take_id, {"worker_id": self.worker_id, "status": decision.status, "run_ref": decision.run_ref, "pipeline_id": decision.pipeline_id})
                self._touch_done_marker(take_id)
                self._emit("RGB_PROCESSED", f"Processed RGB take {take_id}.", metadata={"take_id": take_id, "status": decision.status})
            else:
                errors += 1
                retry = self.queue.mark_failed(take_id, {"worker_id": self.worker_id, "warning": decision.warning, "status": decision.status})
                if retry.get("retried"):
                    retries += 1
                self._emit("RGB_PROCESSING_FAILED", f"Failed RGB processing for {take_id}: {decision.warning or decision.status}", severity="error", metadata={"take_id": take_id, "status": decision.status, "retry": retry})

        return {
            "processed_count": processed,
            "skipped_count": skipped,
            "error_count": errors,
            "retry_count": retries,
            "queue_depth": self.queue.queue_depth_for(modality="rgb"),
            "activity": activity,
        }


class HeightmapAcquisitionProcessingWorker(RuntimeWorkerBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(worker_type="heightmap_acquisition_processing_worker", **kwargs)
        self.manager.register_worker(
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            source_id=self.source_id,
            station_id=self.station_id,
            poll_interval_sec=self.poll_interval_sec,
            enabled=True,
            config={"pipeline_id": "mining_steel_ball_classification_25d", "modality": "heightmap"},
        )

    def run_iteration(self, *, dry_run: bool) -> dict[str, Any]:
        processed = 0
        skipped = 0
        errors = 0
        retries = 0
        activity = False
        stale = self.queue.recover_stale_claims(claim_timeout_sec=180.0)
        if stale:
            self._emit("STALE_CLAIMS_RECOVERED", f"Recovered stale claims: {len(stale)}", metadata={"take_ids": stale})
            activity = True

        for take_id, metadata in _iter_incoming_take_metadata(self.settings.data_dir):
            modalities = [str(item) for item in (metadata.get("modalities") or [])]
            modality = _resolve_25d_modality(modalities)
            if modality is None:
                continue
            if _already_processed_take(self.settings.data_dir, take_id, pipeline_family="25d") or self._has_done_marker(take_id):
                continue
            acquisition_group_id = _ensure_take_group(self.settings, take_id, metadata, station_id=self.station_id)
            source_id = self.source_id or str(metadata.get("source") or "") or "trispector_ftp_0"
            queued = self.queue.enqueue(
                {
                    "take_id": take_id,
                    "source_id": source_id,
                    "modality": "heightmap",
                    "modality_family": "25d",
                    "purpose": "acquisition_inspection",
                    "acquisition_group_id": acquisition_group_id,
                    "frameset_id": metadata.get("frameset", {}).get("frameset_id") if isinstance(metadata.get("frameset"), dict) else None,
                    "capture_timestamp": metadata.get("created_at"),
                    "metadata": {"worker_id": self.worker_id},
                    "max_retries": 3,
                }
            )
            if queued:
                activity = True

        while True:
            claim = self.queue.claim_next(worker_id=self.worker_id, modality="heightmap", purpose="acquisition_inspection", claim_timeout_sec=180.0)
            if claim is None:
                break
            take_id = str(claim.get("take_id") or "")
            activity = True
            if dry_run:
                self.queue.release_claim(take_id)
                skipped += 1
                self._emit("25D_WOULD_PROCESS", f"Dry-run: would process {take_id}", metadata={"take_id": take_id})
                continue
            decision = process_acquired_take_if_bound(
                self.settings,
                AcquiredTakeReady(
                    take_id=take_id,
                    source_id=str(claim.get("source_id") or self.source_id or "trispector_ftp_0"),
                    modality="heightmap",
                    modality_family="25d",
                    acquisition_group_id=str(claim.get("acquisition_group_id") or "") or None,
                    metadata={"worker_id": self.worker_id},
                ),
                purpose="acquisition_inspection",
                auto_process=True,
            )
            if decision.ok and decision.status in {"processing_completed", "processing_binding_missing"}:
                processed += 1
                self.queue.mark_completed(take_id, {"worker_id": self.worker_id, "status": decision.status, "run_ref": decision.run_ref, "pipeline_id": decision.pipeline_id})
                self._touch_done_marker(take_id)
                self._emit("25D_PROCESSED", f"Processed 2.5D take {take_id}.", metadata={"take_id": take_id, "status": decision.status})
            else:
                errors += 1
                retry = self.queue.mark_failed(take_id, {"worker_id": self.worker_id, "warning": decision.warning, "status": decision.status})
                if retry.get("retried"):
                    retries += 1
                self._emit("25D_PROCESSING_FAILED", f"Failed 2.5D processing for {take_id}: {decision.warning or decision.status}", severity="error", metadata={"take_id": take_id, "status": decision.status, "retry": retry})

        return {
            "processed_count": processed,
            "skipped_count": skipped,
            "error_count": errors,
            "retry_count": retries,
            "queue_depth": self.queue.queue_depth_for(modality="heightmap"),
            "activity": activity,
        }


class FusionPublisherWorker(RuntimeWorkerBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(worker_type="fusion_publisher_worker", **kwargs)

    def run_iteration(self, *, dry_run: bool) -> dict[str, Any]:
        processed = 0
        skipped = 0
        errors = 0
        fusion_service = FusionService(self.settings)
        publish_service = PublishedInspectionResultService(self.settings.data_dir)
        bindings = ProcessBindingService(self.settings)

        for group in _iter_groups(self.settings.data_dir):
            group_id = str(group.get("id") or "")
            if not group_id:
                continue
            has_2d = _group_has_candidate_result(self.settings, group_id, "2d")
            has_25d = _group_has_candidate_result(self.settings, group_id, "25d")
            if not has_2d or not has_25d:
                skipped += 1
                continue

            latest_fusion = fusion_service.latest_fusion_for_group(group_id)
            latest_published = publish_service.latest_for_acquisition_group(group_id)
            if _group_already_published_for_latest_fusion(self.settings, group_id, latest_fusion, latest_published):
                skipped += 1
                continue

            source_candidates = [group_id]
            station_id = str(group.get("station_id") or "") or self.station_id
            if station_id:
                source_candidates.append(station_id)
            resolved_binding = None
            for source_id in source_candidates:
                resolved_binding = bindings.resolve_process_binding(source_id, "acquisition_group", "fusion")
                if resolved_binding is not None:
                    break
            if resolved_binding is None:
                skipped += 1
                continue
            recipe_version_id = str(resolved_binding.get("active_recipe_version_id") or "") or None
            if dry_run:
                skipped += 1
                continue
            try:
                fusion_payload = fusion_service.run_fusion(acquisition_group_id=group_id, recipe_version_id=recipe_version_id, force=False, rules={})
                published_payload = publish_service.publish_from_fusion_result(fusion_payload, persist=True)
                processed += 1
                self._emit("FUSION_PUBLISHED", f"Fused and published group {group_id}.", metadata={"group_id": group_id, "fusion_run_id": fusion_payload.get("fusion_run_id"), "published_result_id": published_payload.get("published_result_id")})
            except Exception as exc:  # noqa: BLE001
                errors += 1
                self._emit("FUSION_PUBLISH_FAILED", f"Failed fusion/publish for group {group_id}: {exc}", severity="error", metadata={"group_id": group_id})
        return {"processed_count": processed, "skipped_count": skipped, "error_count": errors, "queue_depth": 0, "activity": bool(processed or skipped)}


class AcquisitionPublicationWorker(RuntimeWorkerBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(worker_type="acquisition_publication_worker", **kwargs)
        self.publisher = PublishedInspectionResultServiceLite(self.settings.data_dir)

    def run_iteration(self, *, dry_run: bool) -> dict[str, Any]:
        processed = 0
        skipped = 0
        errors = 0
        completed_rows = self.queue._read_jsonl(self.queue.completed_path)
        for row in completed_rows[-200:]:
            take_id = str(row.get("take_id") or "")
            if not take_id:
                continue
            marker = self.settings.data_dir / "runtime" / "workers" / "publication" / f"{take_id}.done"
            if marker.is_file():
                skipped += 1
                continue
            result_path = self.settings.processed_dir / take_id / "result.json"
            if not result_path.is_file():
                skipped += 1
                continue
            try:
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                errors += 1
                continue
            if not isinstance(result_payload, dict):
                errors += 1
                continue
            if dry_run:
                skipped += 1
                continue
            try:
                self.publisher.publish_from_take(
                    take_id=take_id,
                    acquisition_group_id=str(row.get("acquisition_group_id") or "") or None,
                    source_id=str(row.get("source_id") or "") or None,
                    result_payload=result_payload,
                )
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                processed += 1
                self._emit("PUBLICATION_CREATED", f"Published operator summary for {take_id}.", metadata={"take_id": take_id})
            except Exception as exc:  # noqa: BLE001
                errors += 1
                self._emit("PUBLICATION_FAILED", f"Failed publishing {take_id}: {exc}", severity="error", metadata={"take_id": take_id})
        return {"processed_count": processed, "skipped_count": skipped, "error_count": errors, "queue_depth": 0, "activity": bool(processed or skipped)}


def _worker_done_marker(data_dir: Path, worker_id: str, take_id: str) -> Path:
    return data_dir / "runtime" / "workers" / "markers" / worker_id / f"{take_id}.done"


def _touch_worker_done_marker(data_dir: Path, worker_id: str, take_id: str) -> None:
    marker = _worker_done_marker(data_dir, worker_id, take_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def _has_worker_done_marker(data_dir: Path, worker_id: str, take_id: str) -> bool:
    return _worker_done_marker(data_dir, worker_id, take_id).is_file()


def _iter_incoming_take_metadata(data_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    incoming = data_dir / "incoming"
    if not incoming.is_dir():
        return []
    rows: list[tuple[str, dict[str, Any]]] = []
    for take_dir in sorted(incoming.iterdir(), key=lambda path: path.name):
        if not take_dir.is_dir() or take_dir.name.startswith("."):
            continue
        path = take_dir / "metadata.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        take_id = str(payload.get("take_id") or take_dir.name)
        rows.append((take_id, payload))
    return rows


def _resolve_25d_modality(modalities: list[str]) -> str | None:
    if "heightmap" in modalities:
        return "heightmap"
    if "derived_25d" in modalities:
        return "derived_25d"
    if "heightmap_2_5d" in modalities:
        return "heightmap_2_5d"
    return None


def _already_processed_take(data_dir: Path, take_id: str, *, pipeline_family: str) -> bool:
    status = read_status(data_dir, take_id)
    if isinstance(status, dict) and str(status.get("status") or "") == "processing_completed":
        return True
    result_path = data_dir / "processed" / take_id / "result.json"
    if not result_path.is_file():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    status_value = str(payload.get("status") or "")
    family = str(((payload.get("processing_pipeline") or {}) if isinstance(payload.get("processing_pipeline"), dict) else {}).get("pipeline_family") or "")
    return family == pipeline_family and status_value in OK_STATUSES


def _ensure_take_group(settings: ApiSettings, take_id: str, metadata: dict[str, Any], *, station_id: str | None) -> str:
    group_id = str(metadata.get("acquisition_group_id") or "")
    groups = AcquisitionGroupService(settings.data_dir)
    if group_id:
        return group_id
    group = groups.create_acquisition_group(
        name=f"auto_{take_id}",
        station_id=station_id,
        metadata={"auto_created_by": "runtime_worker", "take_id": take_id},
    )
    groups.attach_take_to_group(take_id, group["id"])
    return group["id"]


def _iter_groups(data_dir: Path) -> list[dict[str, Any]]:
    groups_dir = data_dir / "acquisition_groups" / "groups"
    if not groups_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(groups_dir.glob("*.json"), key=lambda p: p.name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _group_has_candidate_result(settings: ApiSettings, group_id: str, family: str) -> bool:
    takes = AcquisitionGroupService(settings.data_dir).list_group_takes(group_id)
    for row in takes:
        take_id = str(row.get("take_id") or "")
        if not take_id:
            continue
        result_path = settings.processed_dir / take_id / "result.json"
        if not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        pipeline_family = str(((payload.get("processing_pipeline") or {}) if isinstance(payload.get("processing_pipeline"), dict) else {}).get("pipeline_family") or "")
        status = str(payload.get("status") or "")
        object_candidates = payload.get("object_candidates")
        if pipeline_family == family and status in OK_STATUSES and isinstance(object_candidates, list) and len(object_candidates) > 0:
            return True
    return False


def _group_latest_processed_mtime(settings: ApiSettings, group_id: str) -> float:
    latest = 0.0
    takes = AcquisitionGroupService(settings.data_dir).list_group_takes(group_id)
    for row in takes:
        take_id = str(row.get("take_id") or "")
        if not take_id:
            continue
        result_path = settings.processed_dir / take_id / "result.json"
        if result_path.is_file():
            latest = max(latest, result_path.stat().st_mtime)
    return latest


def _group_already_published_for_latest_fusion(
    settings: ApiSettings,
    group_id: str,
    latest_fusion: dict[str, Any] | None,
    latest_published: dict[str, Any] | None,
) -> bool:
    if latest_fusion is None or latest_published is None:
        return False
    if str(latest_published.get("fusion_run_id") or "") != str(latest_fusion.get("fusion_run_id") or ""):
        return False
    fusion_run_id = str(latest_fusion.get("fusion_run_id") or "")
    if not fusion_run_id:
        return False
    fusion_result_path = settings.data_dir / "fusion" / "runs" / fusion_run_id / "fusion_result.json"
    if not fusion_result_path.is_file():
        return False
    group_updated_at = _group_latest_processed_mtime(settings, group_id)
    return group_updated_at <= fusion_result_path.stat().st_mtime
