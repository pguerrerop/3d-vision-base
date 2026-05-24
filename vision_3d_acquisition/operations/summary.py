from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.operations.classification_superclass import (
    SUPERCLASS_UNKNOWN,
    select_dominant_classification,
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


def extract_classification_variables(obj: dict[str, Any] | None) -> dict[str, float | None]:
    item = obj if isinstance(obj, dict) else {}
    height_metrics = item.get("height_above_belt_mm") if isinstance(item.get("height_above_belt_mm"), dict) else {}

    def as_float(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "max_height_mm": as_float(height_metrics.get("max_height_mm")),
        "p95_height_mm": as_float(height_metrics.get("p95_height_mm")),
        "eccentricity": as_float(item.get("feature_eccentricity")),
        "sphericity_3d": as_float(item.get("feature_sphericity_3d")),
        "flatness": as_float(item.get("feature_flatness")),
        "edge_roughness": as_float(item.get("feature_edge_roughness")),
        "footprint_roundness": as_float(item.get("feature_footprint_roundness") or item.get("feature_sphere_fit")),
        "volume_proxy_mm3": as_float(item.get("feature_volume_proxy_mm3")),
    }


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
    payload = result_payload or {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    canonical = select_dominant_classification(payload)
    label = str(summary.get("label") or canonical.get("label") or "unknown")
    superclass = str(summary.get("superclass") or canonical.get("superclass") or SUPERCLASS_UNKNOWN)
    confidence_raw = summary.get("confidence")
    if not isinstance(confidence_raw, (int, float)):
        confidence_raw = canonical.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None

    status = _status_for_operation(result_payload, runtime_state)
    object_count = int(canonical.get("object_count") or 0)
    primary_object = pick_primary_object(payload)
    card: dict[str, Any] = {
        "take_id": take_id,
        "status": status,
        "superclass": superclass,
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
        "classification_variables": extract_classification_variables(primary_object),
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


def reindex_recent_operations_cards(data_dir: Path, *, limit: int = 200) -> dict[str, Any]:
    incoming_root = data_dir / "incoming"
    processed_root = data_dir / "processed"
    rows: list[tuple[float, str]] = []
    for processed_dir in processed_root.iterdir() if processed_root.is_dir() else []:
        if not processed_dir.is_dir() or processed_dir.name.startswith("."):
            continue
        result_path = processed_dir / "result.json"
        result_payload = read_json(result_path)
        if not isinstance(result_payload, dict):
            continue
        processed_at = str(result_payload.get("processed_at") or "")
        ts = 0.0
        try:
            if processed_at:
                ts = datetime.fromisoformat(processed_at.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = result_path.stat().st_mtime
        rows.append((ts, processed_dir.name))
    rows.sort(key=lambda item: item[0], reverse=True)

    count = 0
    for _, take_id in rows[: max(1, int(limit))]:
        card = build_operations_card(
            take_id=take_id,
            incoming_metadata=read_json(incoming_root / take_id / "metadata.json"),
            runtime_state=read_json(incoming_root / take_id / "runtime_state.json"),
            result_payload=read_json(processed_root / take_id / "result.json"),
            processed_dir=processed_root / take_id,
        )
        write_operations_card(processed_root / take_id, card)
        update_operations_index(data_dir, card, limit=500)
        count += 1
    return {"reindexed": count, "limit": max(1, int(limit))}
