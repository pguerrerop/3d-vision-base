from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


WorkerState = Literal["idle", "running", "stopped", "error"]
WorkerSeverity = Literal["debug", "info", "warning", "error"]


def worker_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class WorkerDefinition:
    worker_id: str
    worker_type: str
    source_id: str | None = None
    station_id: str | None = None
    poll_interval_sec: float = 1.0
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "source_id": self.source_id,
            "station_id": self.station_id,
            "poll_interval_sec": self.poll_interval_sec,
            "enabled": self.enabled,
            "config": self.config,
        }


@dataclass(slots=True)
class WorkerStatus:
    worker_id: str
    worker_type: str
    state: WorkerState
    status: WorkerState | None = None
    source_id: str | None = None
    pipeline_id: str | None = None
    modality: str | None = None
    station_id: str | None = None
    queue_depth: int = 0
    poll_interval_sec: float = 1.0
    started_at: str | None = None
    last_heartbeat: str | None = None
    last_activity_at: str | None = None
    last_success_at: str | None = None
    last_event_at: str | None = None
    last_event_summary: str | None = None
    last_error: str | None = None
    error_count: int = 0
    processed_count: int = 0
    retry_count: int = 0
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "state": self.state,
            "status": self.status or self.state,
            "source_id": self.source_id,
            "pipeline_id": self.pipeline_id,
            "modality": self.modality,
            "station_id": self.station_id,
            "queue_depth": self.queue_depth,
            "poll_interval_sec": self.poll_interval_sec,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "last_activity_at": self.last_activity_at,
            "last_success_at": self.last_success_at,
            "last_event_at": self.last_event_at,
            "last_event_summary": self.last_event_summary,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "processed_count": self.processed_count,
            "retry_count": self.retry_count,
            "stop_requested": self.stop_requested,
        }


@dataclass(slots=True)
class WorkerHeartbeat:
    worker_id: str
    timestamp: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "details": self.details,
        }


@dataclass(slots=True)
class WorkerEvent:
    worker_id: str
    timestamp: str
    severity: WorkerSeverity
    event_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "event_type": self.event_type,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class WorkerRunSummary:
    worker_id: str
    started_at: str
    finished_at: str | None = None
    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "notes": self.notes,
        }


@dataclass(slots=True)
class WorkerHealth:
    worker_id: str
    worker_type: str
    status: WorkerState
    queue_depth: int = 0
    last_activity_at: str | None = None
    last_success_at: str | None = None
    error_count: int = 0
    processed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "status": self.status,
            "queue_depth": self.queue_depth,
            "last_activity_at": self.last_activity_at,
            "last_success_at": self.last_success_at,
            "error_count": self.error_count,
            "processed_count": self.processed_count,
        }
