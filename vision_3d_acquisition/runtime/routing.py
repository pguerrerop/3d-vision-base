from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.datasets import DatasetService

RUNTIME_LIVE_ROUTING_PATH = Path("runtime") / "live_state" / "routing.json"
RUNTIME_DEFAULT_ROUTING_PATH = Path("config") / "runtime_routing.json"


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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class RuntimeRoutingService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.dataset_service = DatasetService(data_dir)
        self.live_path = data_dir / RUNTIME_LIVE_ROUTING_PATH
        self.default_path = Path("config") / "runtime_routing.json"

    def get_live_state(self) -> dict[str, Any] | None:
        return _read_json(self.live_path)

    def set_live_state(
        self,
        *,
        dataset_id: str,
        session_id: str,
        mode: str = "explicit_acquisition_destination",
        source: str = "runtime_ui",
        updated_by: str = "runtime_ui",
    ) -> dict[str, Any]:
        payload = {
            "routing_mode": mode,
            "active_dataset_id": dataset_id,
            "active_session_id": session_id,
            "source": source,
            "updated_at": _now_iso(),
            "updated_by": updated_by,
        }
        _write_json(self.live_path, payload)
        return payload

    def clear_live_state(self) -> bool:
        if not self.live_path.is_file():
            return False
        self.live_path.unlink(missing_ok=True)
        return True

    def get_default_destination(self) -> dict[str, Any] | None:
        return _read_json(self.default_path)

    def set_default_destination(
        self,
        *,
        default_dataset_id: str,
        default_session_id: str,
        auto_create_session: bool,
        session_naming_policy: str,
        updated_by: str = "runtime_ui",
    ) -> dict[str, Any]:
        payload = {
            "default_dataset_id": default_dataset_id,
            "default_session_id": default_session_id,
            "auto_create_session": bool(auto_create_session),
            "session_naming_policy": session_naming_policy,
            "updated_at": _now_iso(),
            "updated_by": updated_by,
        }
        _write_json(self.default_path, payload)
        return payload

    def clear_default_destination(self) -> bool:
        if not self.default_path.is_file():
            return False
        self.default_path.unlink(missing_ok=True)
        return True

    def resolve_routing(
        self,
        *,
        replay_session: dict[str, Any] | None,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = source_metadata or {}
        live = self.get_live_state() or {}
        default = self.get_default_destination() or {}

        if replay_session and bool(replay_session.get("active")):
            return {
                "routing_mode": "replay_override",
                "active_dataset_id": replay_session.get("dataset_id"),
                "active_session_id": replay_session.get("session_id"),
                "source": "replay_session",
                "updated_at": replay_session.get("started_at"),
                "updated_by": replay_session.get("created_by") or "runtime_ui",
                "ownership_explanation": "Replay override is currently controlling acquisition routing.",
            }

        if live.get("active_dataset_id") and live.get("active_session_id"):
            return {
                **live,
                "ownership_explanation": "Explicit acquisition destination is currently controlling acquisition routing.",
            }

        if metadata.get("dataset_id") and metadata.get("session_id"):
            return {
                "routing_mode": "metadata_driven",
                "active_dataset_id": metadata.get("dataset_id"),
                "active_session_id": metadata.get("session_id"),
                "source": "acquisition_metadata",
                "updated_at": _now_iso(),
                "updated_by": "acquisition_source",
                "ownership_explanation": "Acquisition metadata is currently controlling acquisition routing.",
            }

        if default.get("default_dataset_id") and default.get("default_session_id"):
            return {
                "routing_mode": "persistent_default",
                "active_dataset_id": default.get("default_dataset_id"),
                "active_session_id": default.get("default_session_id"),
                "source": "persistent_default",
                "updated_at": default.get("updated_at"),
                "updated_by": default.get("updated_by") or "runtime_ui",
                "ownership_explanation": "Persistent default destination is currently active because no replay or explicit acquisition destination exists.",
            }

        return {
            "routing_mode": "unassigned",
            "active_dataset_id": None,
            "active_session_id": None,
            "source": "none",
            "updated_at": _now_iso(),
            "updated_by": "runtime_system",
            "ownership_explanation": "No routing destination configured.",
        }
