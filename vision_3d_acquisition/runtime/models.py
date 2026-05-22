from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


RuntimeDesiredState = Literal["running", "stopped"]
RuntimeObservedState = Literal["stopped", "starting", "running", "stopping", "crashed", "failed_startup", "orphaned"]
RuntimeSeverity = Literal["debug", "info", "warning", "error"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class RuntimeProcessDefinition:
    process_id: str
    process_type: str
    command: list[str]
    source_id: str | None = None
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeProcessInstance:
    process_id: str
    process_type: str
    pid: int | None
    desired_state: RuntimeDesiredState
    observed_state: RuntimeObservedState
    status_label: str
    started_at: str | None
    stopped_at: str | None
    restart_count: int
    config: dict[str, Any]
    source_id: str | None
    health: str
    last_heartbeat: str | None
    last_event_summary: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_id": self.process_id,
            "process_type": self.process_type,
            "pid": self.pid,
            "desired_state": self.desired_state,
            "observed_state": self.observed_state,
            "status_label": self.status_label,
            "status": self.status_label,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "restart_count": self.restart_count,
            "config": self.config,
            "source_id": self.source_id,
            "health": self.health,
            "last_heartbeat": self.last_heartbeat,
            "last_event_summary": self.last_event_summary,
        }


@dataclass(slots=True)
class RuntimeProcessEvent:
    timestamp: str
    process_id: str
    severity: RuntimeSeverity
    event_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "process_id": self.process_id,
            "severity": self.severity,
            "event_type": self.event_type,
            "message": self.message,
            "metadata": self.metadata,
        }
