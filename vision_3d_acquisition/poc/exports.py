from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.poc.labels import list_labeled_takes, load_labels


def export_labeled_dataset_summary(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in list_labeled_takes(data_dir):
        take_id = str(payload.get("take_id") or "")
        result = _read_json(data_dir / "processed" / take_id / "result.json") or {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        rows.append(
            {
                "take_id": take_id,
                "labels": payload.get("labels", []),
                "notes": payload.get("notes"),
                "reviewer": payload.get("reviewer"),
                "updated_at": payload.get("updated_at"),
                "processed": (data_dir / "processed" / take_id / "DONE").is_file(),
                "object_count": summary.get("object_count"),
                "ball_count": summary.get("ball_count"),
                "non_ball_count": summary.get("non_ball_count"),
            }
        )
    return rows


def export_object_metrics(data_dir: Path, *, take_id: str | None = None) -> list[dict[str, Any]]:
    take_ids = [take_id] if take_id else _processed_take_ids(data_dir)
    rows: list[dict[str, Any]] = []
    for current_take_id in take_ids:
        result = _read_json(data_dir / "processed" / current_take_id / "result.json") or {}
        labels = load_labels(data_dir, current_take_id) or {}
        for status, objects in (("accepted", result.get("objects") or []), ("rejected", result.get("rejected_objects") or [])):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                rows.append(_object_row(current_take_id, status, item, labels))
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _object_row(take_id: str, status: str, item: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    dimensions = item.get("dimensions_mm") or [None, None, None]
    bbox_min = item.get("bbox_min_mm") or [None, None, None]
    bbox_max = item.get("bbox_max_mm") or [None, None, None]
    return {
        "take_id": take_id,
        "take_labels": "|".join(labels.get("labels", [])),
        "object_id": item.get("object_id"),
        "status": status,
        "class_name": item.get("class_name"),
        "confidence": item.get("confidence"),
        "diameter_mm": item.get("diameter_mm") or item.get("diameter_estimate_mm"),
        "sphere_ellipse_fit_error": item.get("fit_rmse_mm"),
        "sphericity_score": item.get("sphericity_score"),
        "point_count": item.get("point_count"),
        "bbox_min_x_mm": _at(bbox_min, 0),
        "bbox_min_y_mm": _at(bbox_min, 1),
        "bbox_min_z_mm": _at(bbox_min, 2),
        "bbox_max_x_mm": _at(bbox_max, 0),
        "bbox_max_y_mm": _at(bbox_max, 1),
        "bbox_max_z_mm": _at(bbox_max, 2),
        "dimension_x_mm": _at(dimensions, 0),
        "dimension_y_mm": _at(dimensions, 1),
        "dimension_z_mm": _at(dimensions, 2),
        "filter_status": item.get("filter_status"),
        "filter_reason": item.get("filter_reason"),
        "fraction_points_inside_belt": item.get("fraction_points_inside_belt"),
    }


def _processed_take_ids(data_dir: Path) -> list[str]:
    processed = data_dir / "processed"
    if not processed.is_dir():
        return []
    return sorted((path.name for path in processed.iterdir() if (path / "result.json").is_file()), reverse=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _at(values: Any, index: int) -> Any:
    return values[index] if isinstance(values, (list, tuple)) and len(values) > index else None
