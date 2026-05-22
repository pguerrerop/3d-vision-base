from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PublishedInspectionResultServiceLite:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "published" / "live_inspection_results"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def publish_from_take(self, *, take_id: str, acquisition_group_id: str | None, source_id: str | None, result_payload: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        result_id = f"result_{take_id}_{now.replace(':', '').replace('-', '')}"
        objects = result_payload.get("objects") if isinstance(result_payload.get("objects"), list) else []
        summary = result_payload.get("summary") if isinstance(result_payload.get("summary"), dict) else {}
        dominant = str(summary.get("decision") or "unknown")
        confidence = summary.get("confidence")
        artifacts = result_payload.get("artifacts") if isinstance(result_payload.get("artifacts"), list) else []
        overlay_path = None
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            if str(item.get("kind") or "") in {"overlay", "image"} and str(item.get("path") or ""):
                overlay_path = str(item.get("path"))
                break
        payload = {
            "result_id": result_id,
            "take_id": take_id,
            "acquisition_group_id": acquisition_group_id,
            "source_ids": [source_id] if source_id else [],
            "dominant_class": dominant,
            "object_count": len(objects),
            "confidence_summary": confidence,
            "overlay_artifact": overlay_path,
            "operator_status": "ready",
            "published_at": now,
        }
        result_dir = self.root / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "published_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        index = self._read_index()
        recent = [item for item in index.get("recent", []) if isinstance(item, dict)]
        recent.insert(0, payload)
        index = {"updated_at": now, "latest_result_id": result_id, "recent": recent[:500]}
        self.index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return payload

    def list_results(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._read_index().get("recent")
        if not isinstance(rows, list):
            return []
        return [item for item in rows[: max(1, min(500, int(limit)))] if isinstance(item, dict)]

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"updated_at": None, "latest_result_id": None, "recent": []}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"updated_at": None, "latest_result_id": None, "recent": []}
        return payload if isinstance(payload, dict) else {"updated_at": None, "latest_result_id": None, "recent": []}
