from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {
    "ball",
    "non-ball",
    "partial ball",
    "belt only",
    "calibration issue",
    "noisy scan",
    "occluded",
    "uncertain",
}


def labels_path(data_dir: Path, take_id: str) -> Path:
    return data_dir / "takes" / take_id / "labels.json"


def load_labels(data_dir: Path, take_id: str) -> dict[str, Any] | None:
    path = labels_path(data_dir, take_id)
    if not path.is_file():
        legacy_path = data_dir / "incoming" / take_id / "labels.json"
        path = legacy_path if legacy_path.is_file() else path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_labels(
    data_dir: Path,
    take_id: str,
    labels: list[str],
    *,
    notes: str | None = None,
    reviewer: str | None = None,
) -> dict[str, Any]:
    invalid = sorted(set(labels) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(f"Unsupported labels: {', '.join(invalid)}")
    payload = {
        "take_id": take_id,
        "labels": sorted(set(labels)),
        "notes": notes,
        "reviewer": reviewer,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = labels_path(data_dir, take_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def list_labeled_takes(data_dir: Path) -> list[dict[str, Any]]:
    root = data_dir / "takes"
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in root.glob("*/labels.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return sorted(rows, key=lambda item: str(item.get("updated_at") or item.get("take_id")), reverse=True)
