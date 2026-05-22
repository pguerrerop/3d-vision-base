from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.operations.classification_superclass import (
    SUPERCLASS_UNKNOWN,
    map_label_to_superclass,
)

CARD_FILENAME = "operations_card.json"
INDEX_PATH = Path("runtime") / "operations_index.json"
OK_STATUSES = {"ok", "success", "warning", "completed"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_preview_from_result(result: dict[str, Any], processed_dir: Path) -> str | None:
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    candidates: list[str] = []
    for name in (
        "classification_overlay.png",
        "height_segmentation_overlay.png",
        "normalized_heightmap_preview.png",
        "raw_heightmap_preview.png",
    ):
        candidates.append(name)
    for key in ("overlay", "height_segmentation_overlay", "normalized_heightmap", "input_preview", "heightmap"):
        value = files.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (processed_dir / candidate).is_file():
            return candidate
    return None


def pick_primary_object(result: dict[str, Any]) -> dict[str, Any] | None:
    objects = result.get("objects") if isinstance(result.get("objects"), list) else []
    if not objects:
        return None
    ranked = sorted(
        [item for item in objects if isinstance(item, dict)],
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )
    return ranked[0] if ranked else None


def primary_label(result: dict[str, Any], obj: dict[str, Any] | None) -> str:
    if isinstance(obj, dict):
        for key in ("class_name", "label", "class_label"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        cls = obj.get("classification")
        if isinstance(cls, dict):
            value = cls.get("class_label") or cls.get("label")
            if isinstance(value, str) and value.strip():
                return value.strip()
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    summary_label = summary.get("label") or summary.get("class_label")
    if isinstance(summary_label, str) and summary_label.strip():
        return summary_label.strip()
    return "Unknown"


def _status_for_operation(result_payload: dict[str, Any] | None, runtime_state: dict[str, Any] | None) -> str:
    runtime_status = str((runtime_state or {}).get("state") or "").strip().lower()
    if runtime_status in {"acquired", "queued", "processing", "completed", "failed"}:
        return runtime_status
    result_status = str((result_payload or {}).get("status") or "").strip().lower()
    if result_status in OK_STATUSES:
        return "completed"
    if result_status == "failed":
        return "failed"
    return runtime_status or "acquired"


def build_operations_card(
    *,
    take_id: str,
    incoming_metadata: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
    processed_dir: Path,
) -> dict[str, Any]:
    primary = pick_primary_object(result_payload or {})
    label = primary_label(result_payload or {}, primary)
    confidence = None
    if isinstance(primary, dict) and isinstance(primary.get("confidence"), (int, float)):
        confidence = float(primary["confidence"])
    else:
        summary = (result_payload or {}).get("summary") if isinstance((result_payload or {}).get("summary"), dict) else {}
        if isinstance(summary.get("confidence"), (int, float)):
            confidence = float(summary["confidence"])

    status = _status_for_operation(result_payload, runtime_state)
    object_count = len((result_payload or {}).get("objects") or []) if isinstance((result_payload or {}).get("objects"), list) else 0
    card: dict[str, Any] = {
        "take_id": take_id,
        "status": status,
        "superclass": map_label_to_superclass(label),
        "label": label,
        "confidence": confidence,
        "object_count": object_count,
        "preview_image": resolve_preview_from_result(result_payload or {}, processed_dir),
        "acquired_at": (incoming_metadata or {}).get("created_at"),
        "processed_at": (result_payload or {}).get("processed_at"),
        "created_at": (runtime_state or {}).get("created_at") or now_iso(),
        "updated_at": now_iso(),
        "source": (runtime_state or {}).get("source") or ((incoming_metadata or {}).get("source") if isinstance((incoming_metadata or {}).get("source"), str) else "trispector_ftp"),
        "modalities": ["heightmap"],
        "error": (result_payload or {}).get("error") if isinstance((result_payload or {}).get("error"), str) else (runtime_state or {}).get("error"),
        "pipeline_id": "mining_steel_ball_classification_25d",
    }
    if card["superclass"] == SUPERCLASS_UNKNOWN and label.lower() == "unknown" and status == "completed":
        # completed with unknown label stays explicit in Operations
        card["superclass"] = SUPERCLASS_UNKNOWN
    return card


def write_operations_card(processed_dir: Path, card: dict[str, Any]) -> Path:
    path = processed_dir / CARD_FILENAME
    write_json(path, card)
    return path


def update_operations_index(data_dir: Path, card: dict[str, Any], *, limit: int = 500) -> dict[str, Any]:
    index_path = data_dir / INDEX_PATH
    payload = read_json(index_path) or {"updated_at": None, "latest_take_id": None, "cards": []}
    rows = [item for item in payload.get("cards", []) if isinstance(item, dict)]
    remaining = [item for item in rows if str(item.get("take_id") or "") != str(card.get("take_id") or "")]
    remaining.insert(0, card)
    next_payload = {
        "updated_at": now_iso(),
        "latest_take_id": card.get("take_id"),
        "cards": remaining[: max(1, int(limit))],
    }
    write_json(index_path, next_payload)
    return next_payload
