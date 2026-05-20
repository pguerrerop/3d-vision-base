from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in base:
        base = base.replace("--", "-")
    return base or f"id-{int(datetime.now(UTC).timestamp())}"


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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class DatasetService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.datasets_dir = data_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for folder in self.datasets_dir.glob("dataset_*"):
            if not folder.is_dir():
                continue
            payload = _read_json(folder / "dataset.json")
            if payload:
                items.append(payload)
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        return _read_json(self._dataset_dir(dataset_id) / "dataset.json")

    def create_dataset(self, *, name: str, description: str | None = None, tags: list[str] | None = None, notes: str | None = None, dataset_id: str | None = None) -> dict[str, Any]:
        did = dataset_id or _slug(name)
        payload = {
            "id": did,
            "name": name,
            "description": description,
            "created_at": _now_iso(),
            "tags": tags or [],
            "notes": notes,
        }
        _write_json(self._dataset_dir(did) / "dataset.json", payload)
        return payload

    def update_dataset(self, dataset_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get_dataset(dataset_id)
        if not current:
            return None
        merged = {**current, **{k: v for k, v in updates.items() if v is not None or k == "notes"}}
        _write_json(self._dataset_dir(dataset_id) / "dataset.json", merged)
        return merged

    def list_sessions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        datasets = [dataset_id] if dataset_id else [item.get("id") for item in self.list_datasets()]
        for did in datasets:
            if not isinstance(did, str) or not did:
                continue
            sessions_dir = self._dataset_dir(did) / "sessions"
            if not sessions_dir.is_dir():
                continue
            for folder in sessions_dir.glob("session_*"):
                payload = _read_json(folder / "session.json")
                if payload:
                    sessions.append(payload)
        return sorted(sessions, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def get_session(self, dataset_id: str, session_id: str) -> dict[str, Any] | None:
        return _read_json(self._session_dir(dataset_id, session_id) / "session.json")

    def create_session(
        self,
        *,
        dataset_id: str,
        name: str,
        description: str | None = None,
        calibration_id: str | None = None,
        sensor_metadata: dict[str, Any] | None = None,
        conveyor_metadata: dict[str, Any] | None = None,
        lighting_metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        sid = session_id or _slug(name)
        payload = {
            "id": sid,
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "calibration_id": calibration_id,
            "sensor_metadata": sensor_metadata or {},
            "conveyor_metadata": conveyor_metadata or {},
            "lighting_metadata": lighting_metadata or {},
            "created_at": _now_iso(),
            "tags": tags or [],
            "notes": notes,
        }
        _write_json(self._session_dir(dataset_id, sid) / "session.json", payload)
        return payload

    def update_session(self, dataset_id: str, session_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get_session(dataset_id, session_id)
        if not current:
            return None
        merged = {**current, **{k: v for k, v in updates.items() if v is not None or k == "notes"}}
        _write_json(self._session_dir(dataset_id, session_id) / "session.json", merged)
        return merged

    def resolve_take_membership(self, take_id: str) -> tuple[str | None, str | None]:
        for dataset in self.list_datasets():
            did = str(dataset.get("id") or "")
            if not did:
                continue
            sessions_dir = self._dataset_dir(did) / "sessions"
            if not sessions_dir.is_dir():
                continue
            for session_folder in sessions_dir.glob("session_*"):
                sid = session_folder.name.replace("session_", "", 1)
                take_folder = session_folder / "takes" / take_id
                if take_folder.is_dir():
                    return did, sid
        return None, None

    def upsert_take_metadata(
        self,
        *,
        take_id: str,
        dataset_id: str,
        session_id: str,
        updates: dict[str, Any],
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        take_path = self._session_dir(dataset_id, session_id) / "takes" / take_id / "metadata.json"
        existing = _read_json(take_path) or self.default_take_metadata(take_id, source_metadata)
        merged = {**existing, **updates, "take_id": take_id, "dataset_id": dataset_id, "session_id": session_id}
        _write_json(take_path, merged)
        return merged

    def load_take_metadata(self, *, take_id: str, source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        did, sid = self.resolve_take_membership(take_id)
        if did and sid:
            payload = _read_json(self._session_dir(did, sid) / "takes" / take_id / "metadata.json")
            if payload:
                payload.setdefault("dataset_id", did)
                payload.setdefault("session_id", sid)
                return payload
        fallback = self.default_take_metadata(take_id, source_metadata)
        if did:
            fallback["dataset_id"] = did
        if sid:
            fallback["session_id"] = sid
        return fallback

    def default_take_metadata(self, take_id: str, source_metadata: dict[str, Any] | None) -> dict[str, Any]:
        source = source_metadata or {}
        return {
            "take_id": take_id,
            "friendly_name": take_id,
            "labels": [],
            "tags": [],
            "notes": None,
            "expected_class": None,
            "expected_diameter_mm": None,
            "expected_count": None,
            "operator_notes": None,
            "validation_status": "unreviewed",
            "dataset_id": None,
            "session_id": source.get("session_id"),
            "created_at": source.get("created_at") or _now_iso(),
            "object_annotations": [],
        }

    def upsert_object_annotation(
        self,
        *,
        take_id: str,
        dataset_id: str,
        session_id: str,
        annotation: dict[str, Any],
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.upsert_take_metadata(
            take_id=take_id,
            dataset_id=dataset_id,
            session_id=session_id,
            updates={},
            source_metadata=source_metadata,
        )
        current = payload.get("object_annotations")
        annotations = [item for item in current if isinstance(item, dict)] if isinstance(current, list) else []
        annotation_id = str(annotation.get("id") or "").strip() or f"object_{len(annotations) + 1:03d}"
        now = _now_iso()
        incoming = {
            "id": annotation_id,
            "source_stage": annotation.get("source_stage"),
            "source_artifact_id": annotation.get("source_artifact_id"),
            "candidate_id": str(annotation.get("candidate_id") or ""),
            "bbox": annotation.get("bbox"),
            "centroid": annotation.get("centroid"),
            "contour_ref": annotation.get("contour_ref"),
            "labels": [str(v) for v in (annotation.get("labels") or []) if str(v)],
            "expected_class": annotation.get("expected_class"),
            "expected_diameter_mm": annotation.get("expected_diameter_mm"),
            "notes": annotation.get("notes"),
            "validation_status": annotation.get("validation_status") or "unreviewed",
            "created_at": annotation.get("created_at") or now,
            "updated_at": now,
        }
        updated = False
        next_annotations: list[dict[str, Any]] = []
        for existing in annotations:
            if str(existing.get("id") or "") == annotation_id:
                merged = {**existing, **incoming}
                if existing.get("created_at"):
                    merged["created_at"] = existing.get("created_at")
                next_annotations.append(merged)
                updated = True
            else:
                next_annotations.append(existing)
        if not updated:
            next_annotations.append(incoming)
        payload["object_annotations"] = next_annotations
        take_path = self._session_dir(dataset_id, session_id) / "takes" / take_id / "metadata.json"
        _write_json(take_path, payload)
        return incoming

    def match_object_annotations(
        self,
        *,
        annotations: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not annotations:
            return []
        by_candidate_id = {str(c.get("candidate_id") or ""): c for c in candidates if str(c.get("candidate_id") or "")}
        matched: list[dict[str, Any]] = []
        for annotation in annotations:
            item = dict(annotation)
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id and candidate_id in by_candidate_id:
                item["matched_candidate_id"] = candidate_id
                item["matched_by"] = "candidate_id"
                matched.append(item)
                continue
            resolved = _match_by_geometry(item, candidates)
            item["matched_candidate_id"] = resolved.get("candidate_id") if resolved else None
            item["matched_by"] = resolved.get("matched_by") if resolved else None
            matched.append(item)
        return matched

    def _dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / f"dataset_{dataset_id}"

    def _session_dir(self, dataset_id: str, session_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "sessions" / f"session_{session_id}"


def _match_by_geometry(annotation: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    ann_bbox = _quad(annotation.get("bbox"))
    ann_centroid = _pair(annotation.get("centroid"))
    if ann_bbox is not None:
        best_iou = 0.0
        best: dict[str, Any] | None = None
        for candidate in candidates:
            bbox = _quad(candidate.get("bbox"))
            if bbox is None:
                continue
            iou = _bbox_iou(ann_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best = candidate
        if best is not None and best_iou >= 0.2:
            return {**best, "matched_by": "bbox_iou"}
    if ann_centroid is not None:
        nearest: dict[str, Any] | None = None
        best_dist = float("inf")
        for candidate in candidates:
            centroid = _pair(candidate.get("centroid"))
            if centroid is None:
                continue
            distance = math.dist(ann_centroid, centroid)
            if distance < best_dist:
                best_dist = distance
                nearest = candidate
        if nearest is not None:
            return {**nearest, "matched_by": "nearest_centroid"}
    return None


def _pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _quad(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    except Exception:
        return None


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = (aw * ah) + (bw * bh) - intersection
    return (intersection / union) if union > 0 else 0.0
