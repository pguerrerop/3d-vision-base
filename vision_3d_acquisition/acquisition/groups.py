from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.datasets import DatasetService

class AcquisitionGroupService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.root = data_dir / "acquisition_groups"
        self.groups_dir = self.root / "groups"
        self.groups_dir.mkdir(parents=True, exist_ok=True)

    def create_acquisition_group(
        self,
        *,
        name: str | None = None,
        station_id: str | None = None,
        trigger_id: str | None = None,
        encoder_position: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now_iso()
        group_id = f"ag_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        payload = {
            "id": group_id,
            "name": name,
            "station_id": station_id,
            "trigger_id": trigger_id,
            "encoder_position": encoder_position,
            "started_at": now,
            "completed_at": None,
            "status": "open",
            "metadata": metadata or {},
        }
        self._write_group(payload)
        return payload

    def get_acquisition_group(self, group_id: str) -> dict[str, Any] | None:
        payload = _read_json(self.groups_dir / f"{group_id}.json")
        return payload if isinstance(payload, dict) else None

    def update_acquisition_group(self, group_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.get_acquisition_group(group_id)
        if not current:
            raise KeyError(f"Unknown acquisition group: {group_id}")
        merged = {**current, **patch}
        if merged.get("status") == "complete" and not merged.get("completed_at"):
            merged["completed_at"] = _now_iso()
        self._write_group(merged)
        return merged

    def attach_take_to_group(self, take_id: str, acquisition_group_id: str) -> dict[str, Any]:
        group = self.get_acquisition_group(acquisition_group_id)
        if not group:
            raise KeyError(f"Unknown acquisition group: {acquisition_group_id}")
        take_dir = self.data_dir / "incoming" / take_id
        metadata_path = take_dir / "metadata.json"
        if not metadata_path.is_file():
            raise KeyError(f"Take not found: {take_id}")
        payload = _read_json(metadata_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid take metadata for: {take_id}")
        payload["acquisition_group_id"] = acquisition_group_id
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        dataset_service = DatasetService(self.data_dir)
        did, sid = dataset_service.resolve_take_membership(take_id)
        if did and sid:
            dataset_service.upsert_take_metadata(
                take_id=take_id,
                dataset_id=did,
                session_id=sid,
                updates={"acquisition_group_id": acquisition_group_id},
                source_metadata=payload,
            )
        return payload

    def list_group_takes(self, group_id: str) -> list[dict[str, Any]]:
        group = self.get_acquisition_group(group_id)
        if not group:
            raise KeyError(f"Unknown acquisition group: {group_id}")
        incoming = self.data_dir / "incoming"
        if not incoming.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for take_dir in incoming.iterdir():
            if not take_dir.is_dir() or take_dir.name.startswith("."):
                continue
            meta = _read_json(take_dir / "metadata.json")
            if not isinstance(meta, dict):
                continue
            if str(meta.get("acquisition_group_id") or "") != group_id:
                continue
            items.append(
                {
                    "take_id": str(meta.get("take_id") or take_dir.name),
                    "created_at": meta.get("created_at"),
                    "modalities": meta.get("modalities") or [],
                    "source": meta.get("source"),
                    "acquisition_group_id": group_id,
                }
            )
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items

    def _write_group(self, payload: dict[str, Any]) -> None:
        group_id = str(payload.get("id") or "")
        if not group_id:
            raise ValueError("acquisition group id is required")
        path = self.groups_dir / f"{group_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
