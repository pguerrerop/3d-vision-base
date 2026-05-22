from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def status_path(data_dir: Path, take_id: str) -> Path:
    return data_dir / "acquisition_processing" / "status" / f"{take_id}.json"


def read_status(data_dir: Path, take_id: str) -> dict[str, Any] | None:
    path = status_path(data_dir, take_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_status(data_dir: Path, take_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = status_path(data_dir, take_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document.setdefault("take_id", take_id)
    document.setdefault("updated_at", datetime.now(UTC).isoformat())
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def update_status(data_dir: Path, take_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    current = read_status(data_dir, take_id) or {"take_id": take_id, "status": "acquired"}
    current.update(patch)
    current["updated_at"] = datetime.now(UTC).isoformat()
    return write_status(data_dir, take_id, current)
