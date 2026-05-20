from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_runtime_status() -> dict[str, Any]:
    return {
        "status": "idle",
        "latest_take_id": None,
        "latest_processed_take_id": None,
        "message": "Waiting for acquisition and processed output.",
        "updated_at": None,
        "acquisition_connected": False,
        "latest_frame_timestamp": None,
        "processing_lag_ms": None,
        "queue_size": 0,
        "dropped_frames": 0,
        "last_successful_processing": None,
        "current_session": None,
        "active_calibration": None,
        "acquisition_source": None,
        "acquisition_source_details": {},
        "preview_available": False,
        "preview_timestamp": None,
        "preview_fps_estimate": None,
        "preview_stale": True,
        "preview_source": None,
        "preview_path": None,
        "stale": True,
        "throughput": {},
        "warnings": [],
    }


def read_runtime_status(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "runtime.json"
    if not path.is_file():
        return default_runtime_status()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_runtime_status()
    if not isinstance(payload, dict):
        return default_runtime_status()
    merged = default_runtime_status()
    merged.update(payload)
    merged["warnings"] = list(merged.get("warnings") or [])
    merged["throughput"] = dict(merged.get("throughput") or {})
    return merged


def write_runtime_status(state_dir: Path, payload: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "runtime.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_runtime_status(state_dir: Path, **updates: Any) -> dict[str, Any]:
    status = read_runtime_status(state_dir)
    status.update({key: value for key, value in updates.items() if value is not None})
    status["updated_at"] = updates.get("updated_at") or datetime.now(UTC).isoformat()
    status["stale"] = is_stale(status)
    write_runtime_status(state_dir, status)
    return status


def is_stale(status: dict[str, Any], *, threshold_seconds: float = 5.0) -> bool:
    timestamp = status.get("latest_frame_timestamp") or status.get("updated_at")
    if not timestamp:
        return True
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return True
    age = (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds()
    return age > threshold_seconds
