from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.settings import ApiSettings, get_settings
from vision_3d_acquisition.runtime.worker_models import (
    WorkerDefinition,
    WorkerEvent,
    WorkerHeartbeat,
    WorkerStatus,
    worker_now_iso,
)
from vision_3d_acquisition.runtime.processing_queue import TakeProcessingQueue


class RuntimeWorkerManager:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "runtime" / "workers"
        self.definitions_dir = self.root / "definitions"
        self.status_dir = self.root / "status"
        self.events_dir = self.root / "events"
        self.stop_requests_dir = self.root / "stop_requests"
        for path in (self.definitions_dir, self.status_dir, self.events_dir, self.stop_requests_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.queue = TakeProcessingQueue(settings.data_dir)

    def register_worker(
        self,
        *,
        worker_id: str,
        worker_type: str,
        source_id: str | None = None,
        station_id: str | None = None,
        poll_interval_sec: float = 1.0,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            definition = WorkerDefinition(
                worker_id=worker_id,
                worker_type=worker_type,
                source_id=source_id,
                station_id=station_id,
                poll_interval_sec=max(0.1, float(poll_interval_sec)),
                enabled=enabled,
                config=dict(config or {}),
            )
            self._write_json(self._definition_path(worker_id), definition.to_dict())
            current_status = self.get_worker_status(worker_id)
            if current_status is None:
                status = WorkerStatus(
                    worker_id=worker_id,
                    worker_type=worker_type,
                    state="idle",
                    status="idle",
                    source_id=source_id,
                    station_id=station_id,
                    pipeline_id=str((config or {}).get("pipeline_id") or "") or None,
                    modality=str((config or {}).get("modality") or "") or None,
                    poll_interval_sec=definition.poll_interval_sec,
                    started_at=worker_now_iso(),
                    stop_requested=self.is_stop_requested(worker_id),
                )
                self._write_json(self._status_path(worker_id), status.to_dict())
            else:
                merged = dict(current_status)
                merged["worker_type"] = worker_type
                merged["source_id"] = source_id
                merged["station_id"] = station_id
                merged["pipeline_id"] = str((config or {}).get("pipeline_id") or merged.get("pipeline_id") or "") or None
                merged["modality"] = str((config or {}).get("modality") or merged.get("modality") or "") or None
                merged["poll_interval_sec"] = definition.poll_interval_sec
                merged["stop_requested"] = self.is_stop_requested(worker_id)
                self._write_json(self._status_path(worker_id), merged)
            return definition.to_dict()

    def get_worker_status(self, worker_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._read_json(self._status_path(worker_id))
            return payload if isinstance(payload, dict) else None

    def list_worker_statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            statuses: list[dict[str, Any]] = []
            for path in sorted(self.status_dir.glob("*.json")):
                payload = self._read_json(path)
                if isinstance(payload, dict):
                    statuses.append(payload)
            statuses.sort(key=lambda item: str(item.get("worker_id") or ""))
            return statuses

    def append_worker_event(self, worker_id: str, event: WorkerEvent | dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            event_payload = event.to_dict() if isinstance(event, WorkerEvent) else dict(event)
            event_payload.setdefault("worker_id", worker_id)
            event_payload.setdefault("timestamp", worker_now_iso())
            event_payload.setdefault("severity", "info")
            event_payload.setdefault("event_type", "WORKER_EVENT")
            event_payload.setdefault("message", "")
            event_payload.setdefault("metadata", {})
            events_path = self._events_path(worker_id)
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload) + "\n")
            status = self.get_worker_status(worker_id)
            if status:
                status["last_event_at"] = event_payload["timestamp"]
                status["last_activity_at"] = event_payload["timestamp"]
                status["last_event_summary"] = event_payload["message"]
                status["state"] = "error" if event_payload.get("severity") == "error" else status.get("state") or "running"
                status["status"] = status["state"]
                if event_payload.get("severity") == "error":
                    status["error_count"] = int(status.get("error_count") or 0) + 1
                self._write_json(self._status_path(worker_id), status)
            return event_payload

    def heartbeat(self, worker_id: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            heartbeat = WorkerHeartbeat(worker_id=worker_id, timestamp=worker_now_iso(), status=status, details=dict(details or {}))
            current = self.get_worker_status(worker_id)
            if current:
                current["last_heartbeat"] = heartbeat.timestamp
                current["state"] = status if status in {"idle", "running", "stopped", "error"} else "running"
                current["status"] = current["state"]
                details_dict = dict(details or {})
                if "queue_depth" in details_dict:
                    current["queue_depth"] = int(details_dict.get("queue_depth") or 0)
                if "processed_count" in details_dict:
                    current["processed_count"] = int(details_dict.get("processed_count") or 0)
                if "error_count" in details_dict:
                    current["error_count"] = int(details_dict.get("error_count") or 0)
                if "retry_count" in details_dict:
                    current["retry_count"] = int(details_dict.get("retry_count") or 0)
                if bool(details_dict.get("activity", False)):
                    current["last_activity_at"] = heartbeat.timestamp
                if bool(details_dict.get("success", False)):
                    current["last_success_at"] = heartbeat.timestamp
                current["stop_requested"] = self.is_stop_requested(worker_id)
                self._write_json(self._status_path(worker_id), current)
            self.append_worker_event(
                worker_id,
                WorkerEvent(
                    worker_id=worker_id,
                    timestamp=heartbeat.timestamp,
                    severity="debug",
                    event_type="HEARTBEAT",
                    message=f"Heartbeat: {status}",
                    metadata=heartbeat.to_dict(),
                ),
            )
            return heartbeat.to_dict()

    def mark_worker_error(self, worker_id: str, error: str) -> dict[str, Any]:
        with self._lock:
            current = self.get_worker_status(worker_id) or {"worker_id": worker_id}
            current["state"] = "error"
            current["status"] = "error"
            current["last_error"] = str(error)
            current["last_event_at"] = worker_now_iso()
            current["last_activity_at"] = current["last_event_at"]
            current["error_count"] = int(current.get("error_count") or 0) + 1
            current["stop_requested"] = self.is_stop_requested(worker_id)
            self._write_json(self._status_path(worker_id), current)
            event = self.append_worker_event(
                worker_id,
                WorkerEvent(
                    worker_id=worker_id,
                    timestamp=worker_now_iso(),
                    severity="error",
                    event_type="WORKER_ERROR",
                    message=str(error),
                ),
            )
            return {"status": current, "event": event}

    def get_worker_events(self, worker_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            path = self._events_path(worker_id)
            if not path.is_file():
                return []
            lines = path.read_text(encoding="utf-8").splitlines()
            rows: list[dict[str, Any]] = []
            for line in lines[-max(1, min(5000, int(limit))):]:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
            return rows

    def request_stop(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._stop_request_path(worker_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(worker_now_iso(), encoding="utf-8")
            current = self.get_worker_status(worker_id)
            if current:
                current["stop_requested"] = True
                self._write_json(self._status_path(worker_id), current)
            self.append_worker_event(
                worker_id,
                WorkerEvent(
                    worker_id=worker_id,
                    timestamp=worker_now_iso(),
                    severity="info",
                    event_type="STOP_REQUESTED",
                    message="Stop requested by API.",
                ),
            )
            return {"worker_id": worker_id, "stop_requested": True}

    def queue_status(self) -> dict[str, Any]:
        return self.queue.queue_status()

    def is_stop_requested(self, worker_id: str) -> bool:
        return self._stop_request_path(worker_id).is_file()

    def clear_stop_request(self, worker_id: str) -> None:
        path = self._stop_request_path(worker_id)
        if path.is_file():
            path.unlink(missing_ok=True)

    def _definition_path(self, worker_id: str) -> Path:
        return self.definitions_dir / f"{worker_id}.json"

    def _status_path(self, worker_id: str) -> Path:
        return self.status_dir / f"{worker_id}.json"

    def _events_path(self, worker_id: str) -> Path:
        return self.events_dir / f"{worker_id}.jsonl"

    def _stop_request_path(self, worker_id: str) -> Path:
        return self.stop_requests_dir / f"{worker_id}.flag"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


_WORKER_MANAGER: RuntimeWorkerManager | None = None
_WORKER_MANAGER_LOCK = threading.Lock()


def get_runtime_worker_manager() -> RuntimeWorkerManager:
    global _WORKER_MANAGER
    with _WORKER_MANAGER_LOCK:
        if _WORKER_MANAGER is None:
            _WORKER_MANAGER = RuntimeWorkerManager(get_settings())
        return _WORKER_MANAGER
