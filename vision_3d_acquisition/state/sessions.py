from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.contracts.metadata import AcquisitionSessionMetadata


def generate_session_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y_%m_%d_%H%M%S")
    return f"session_{timestamp}"


def ensure_session(
    sessions_dir: Path,
    session_id: str,
    *,
    operator: str | None = None,
    sensor_setup: str | None = None,
    acquisition_mode: str | None = None,
    calibration_used: str | None = None,
    conveyor_speed: float | None = None,
    encoder_enabled: bool | None = None,
    notes: str | None = None,
) -> AcquisitionSessionMetadata:
    session_dir = sessions_dir / session_id
    metadata_path = session_dir / "metadata.json"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "takes").mkdir(parents=True, exist_ok=True)
    if metadata_path.is_file():
        existing = _read_json(metadata_path) or {}
        return AcquisitionSessionMetadata.model_validate(existing)
    metadata = AcquisitionSessionMetadata(
        session_id=session_id,
        operator=operator,
        sensor_setup=sensor_setup,
        acquisition_mode=acquisition_mode,
        calibration_used=calibration_used,
        conveyor_speed=conveyor_speed,
        encoder_enabled=encoder_enabled,
        notes=notes,
    )
    _write_json(metadata_path, metadata.model_dump(mode="json"))
    return metadata


def attach_take_to_session(sessions_dir: Path, session_id: str, take_id: str, take_metadata: dict[str, Any] | None = None) -> None:
    session = ensure_session(sessions_dir, session_id)
    if take_id not in session.take_ids:
        session.take_ids.append(take_id)
        _write_json((sessions_dir / session_id / "metadata.json"), session.model_dump(mode="json"))
    take_dir = sessions_dir / session_id / "takes" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "take_id": take_id,
        "session_id": session_id,
        "attached_at": datetime.now(UTC).isoformat(),
    }
    if take_metadata:
        payload["created_at"] = take_metadata.get("created_at")
        payload["source"] = take_metadata.get("source")
        payload["mode"] = take_metadata.get("mode")
    _write_json(take_dir / "metadata.json", payload)


def list_sessions(sessions_dir: Path) -> list[AcquisitionSessionMetadata]:
    if not sessions_dir.exists():
        return []
    sessions: list[AcquisitionSessionMetadata] = []
    for child in sessions_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        metadata = _read_json(child / "metadata.json")
        if metadata is None:
            continue
        sessions.append(AcquisitionSessionMetadata.model_validate(metadata))
    return sorted(sessions, key=lambda item: item.created_at, reverse=True)


def session_summary(sessions_dir: Path, session_id: str) -> dict[str, Any] | None:
    metadata = _read_json(sessions_dir / session_id / "metadata.json")
    if metadata is None:
        return None
    session = AcquisitionSessionMetadata.model_validate(metadata)
    return {
        "session_id": session.session_id,
        "operator": session.operator,
        "sensor_setup": session.sensor_setup,
        "acquisition_mode": session.acquisition_mode,
        "calibration_used": session.calibration_used,
        "conveyor_speed": session.conveyor_speed,
        "encoder_enabled": session.encoder_enabled,
        "notes": session.notes,
        "created_at": session.created_at,
        "take_count": len(session.take_ids),
        "take_ids": session.take_ids,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
