from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.service import _extract_scalar_features


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PhysicalObjectRegistry:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings.data_dir)

    def list_objects(
        self,
        dataset_id: str,
        *,
        normalized_class: str | None = None,
        superclass: str | None = None,
        session_id: str | None = None,
        needs_review: bool | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        objects_dir = self._objects_dir(dataset_id)
        if not objects_dir.is_dir():
            return []
        for folder in sorted(objects_dir.glob("physical_object_*")):
            payload = _read_json(folder / "physical_object.json")
            if not payload:
                continue
            if normalized_class and str(payload.get("normalized_class") or "") != normalized_class:
                continue
            if superclass and str(payload.get("superclass") or "") != superclass:
                continue
            if session_id and session_id not in [str(item) for item in (payload.get("source_session_ids") or [])]:
                continue
            if needs_review is not None and bool(payload.get("needs_review")) is not bool(needs_review):
                continue
            rows.append(payload)
        return rows

    def get_object(self, dataset_id: str, physical_object_id: str) -> dict[str, Any] | None:
        return _read_json(self._object_dir(dataset_id, physical_object_id) / "physical_object.json")

    def upsert_object(self, dataset_id: str, physical_object_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        current = self.get_object(dataset_id, physical_object_id) or {
            "physical_object_id": physical_object_id,
            "dataset_id": dataset_id,
            "source_session_ids": [],
            "tags": [],
            "notes": "",
            "created_at": _now_iso(),
        }
        merged = {**current, **updates, "physical_object_id": physical_object_id, "dataset_id": dataset_id, "updated_at": _now_iso()}
        source_session_ids = sorted({str(item) for item in (merged.get("source_session_ids") or []) if str(item)})
        merged["source_session_ids"] = source_session_ids
        _write_json(self._object_dir(dataset_id, physical_object_id) / "physical_object.json", merged)
        return merged

    def sync_from_manifest_rows(self, dataset_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            obj_id = str(row.get("physical_object_id") or "").strip()
            if not obj_id:
                continue
            grouped.setdefault(obj_id, []).append(row)
        synced: list[dict[str, Any]] = []
        for obj_id, members in grouped.items():
            first = members[0]
            d_values = [self._to_float(item.get("d1_mm")) for item in members] + [self._to_float(item.get("d2_mm")) for item in members] + [self._to_float(item.get("d3_mm")) for item in members]
            dims = [value for value in d_values if value is not None]
            mean_d = round(sum(dims) / len(dims), 4) if dims else None
            range_d = round(max(dims) - min(dims), 4) if len(dims) >= 2 else 0.0 if dims else None
            payload = self.upsert_object(
                dataset_id,
                obj_id,
                {
                    "source_session_ids": sorted({str(item.get("source_session_id") or "") for item in members if str(item.get("source_session_id") or "")} ),
                    "raw_operator_label": first.get("raw_operator_label"),
                    "normalized_class": first.get("normalized_class"),
                    "superclass": first.get("superclass"),
                    "d1_mm": first.get("d1_mm"),
                    "d2_mm": first.get("d2_mm"),
                    "d3_mm": first.get("d3_mm"),
                    "diameter_mean_mm": mean_d,
                    "diameter_range_mm": range_d,
                    "annotation_confidence": self._confidence_bucket(first.get("annotation_confidence")),
                    "needs_review": any(bool(item.get("needs_review")) for item in members),
                    "source_type": "operator_table",
                    "source_row_index": first.get("source_row_index"),
                    "observation_take_ids": [str(item.get("take_id") or "") for item in members if str(item.get("take_id") or "")],
                },
            )
            synced.append(payload)
        return synced

    def object_takes(self, dataset_id: str, physical_object_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sid, take_id, payload in self.dataset_service.iter_dataset_takes(dataset_id):
            if str(payload.get("physical_object_id") or "") != physical_object_id:
                continue
            rows.append({"session_id": sid, "take_id": take_id, "metadata": payload})
        rows.sort(key=lambda item: str((item.get("metadata") or {}).get("created_at") or ""))
        return rows

    def object_repeatability(self, dataset_id: str, physical_object_id: str) -> dict[str, Any]:
        take_rows = self.object_takes(dataset_id, physical_object_id)
        feature_values: dict[str, list[float]] = {}
        class_votes: dict[str, int] = {}
        for row in take_rows:
            take_id = str(row.get("take_id") or "")
            result = _read_json(self.settings.processed_dir / take_id / "result.json") or {}
            for feature_name, value in _extract_scalar_features(result).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_values.setdefault(feature_name, []).append(float(value))
            take_meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            label = str(take_meta.get("expected_class") or "").strip()
            if label:
                class_votes[label] = class_votes.get(label, 0) + 1
        repeatability = []
        for feature_name, values in sorted(feature_values.items()):
            if not values:
                continue
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            repeatability.append({
                "feature": feature_name,
                "count": len(values),
                "mean": mean_value,
                "variance": variance,
                "min": min(values),
                "max": max(values),
            })
        dominant_class = sorted(class_votes.items(), key=lambda item: (-item[1], item[0]))[0][0] if class_votes else None
        return {
            "physical_object_id": physical_object_id,
            "take_count": len(take_rows),
            "dominant_class": dominant_class,
            "class_votes": class_votes,
            "feature_repeatability": repeatability,
        }

    def validate_object(self, dataset_id: str, physical_object_id: str) -> list[dict[str, Any]]:
        take_rows = self.object_takes(dataset_id, physical_object_id)
        warnings: list[dict[str, Any]] = []
        classes = {str((row.get("metadata") or {}).get("expected_class") or "").strip() for row in take_rows if str((row.get("metadata") or {}).get("expected_class") or "").strip()}
        if len(classes) > 1:
            warnings.append({
                "severity": "warning",
                "code": "inconsistent_normalized_classes",
                "explanation": "Observed takes disagree on normalized class.",
                "affected_count": len(classes),
            })
        if any(bool((row.get("metadata") or {}).get("validation_status") == "needs_review") for row in take_rows):
            warnings.append({
                "severity": "warning",
                "code": "needs_review_labels",
                "explanation": "One or more takes remain review-required.",
                "affected_count": sum(1 for row in take_rows if (row.get("metadata") or {}).get("validation_status") == "needs_review"),
            })
        return warnings

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _confidence_bucket(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "UNKNOWN"
        if numeric >= 0.85:
            return "HIGH"
        if numeric >= 0.55:
            return "MEDIUM"
        return "LOW"

    def _objects_dir(self, dataset_id: str) -> Path:
        return self.settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "physical_objects"

    def _object_dir(self, dataset_id: str, physical_object_id: str) -> Path:
        return self._objects_dir(dataset_id) / f"physical_object_{physical_object_id}"
