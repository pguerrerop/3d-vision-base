from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision_3d_acquisition.runtime.models import RuntimeProcessEvent, utc_now_iso


EVENT_PREFIX = "RUNTIME_EVENT "


@dataclass(slots=True)
class RuntimeLogStore:
    max_lines: int = 2000
    max_events: int = 500
    logs: deque[str] = field(default_factory=deque)
    events: deque[RuntimeProcessEvent] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.logs = deque(maxlen=self.max_lines)
        self.events = deque(maxlen=self.max_events)

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))

    def append_event(self, event: RuntimeProcessEvent) -> None:
        self.events.append(event)

    def tail_logs(self, limit: int = 200) -> list[str]:
        if limit <= 0:
            return []
        return list(self.logs)[-limit:]

    def tail_events(self, limit: int = 200) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return [item.to_dict() for item in list(self.events)[-limit:]]


def encode_event_line(event: RuntimeProcessEvent) -> str:
    return EVENT_PREFIX + json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=True)


def parse_event_line(raw_line: str) -> RuntimeProcessEvent | None:
    line = raw_line.strip()
    if not line.startswith(EVENT_PREFIX):
        return None
    body = line[len(EVENT_PREFIX) :].strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return RuntimeProcessEvent(
        timestamp=str(payload.get("timestamp") or utc_now_iso()),
        process_id=str(payload.get("process_id") or ""),
        severity=str(payload.get("severity") or "info"),  # type: ignore[arg-type]
        event_type=str(payload.get("event_type") or "RUNTIME_EVENT"),
        message=str(payload.get("message") or ""),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )


def write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def read_last_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle.readlines()]
    return lines[-limit:]


def parse_event_json_line(raw_line: str) -> RuntimeProcessEvent | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return RuntimeProcessEvent(
        timestamp=str(payload.get("timestamp") or utc_now_iso()),
        process_id=str(payload.get("process_id") or ""),
        severity=str(payload.get("severity") or "info"),  # type: ignore[arg-type]
        event_type=str(payload.get("event_type") or "RUNTIME_EVENT"),
        message=str(payload.get("message") or ""),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )

