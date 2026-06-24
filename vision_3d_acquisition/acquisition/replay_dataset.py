from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.runtime_config import read_runtime_config


REPLAY_STATE_DIR = Path("runtime") / "acquisition_replay"
ACTIVE_SESSION_FILE = "active_session.json"


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


def _sha256_json(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


@dataclass
class ReplaySessionRef:
    dataset_id: str
    session_id: str


class ReplayableAcquisitionService:
    """Filesystem-first replayable acquisition helper for 2.5D takes."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.dataset_service = DatasetService(data_dir)
        self.state_dir = data_dir / REPLAY_STATE_DIR
        self.active_session_path = self.state_dir / ACTIVE_SESSION_FILE

    def start_session(
        self,
        *,
        dataset_id: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        persist_on_restart: bool = False,
        replay_mode: str = "routing_override",
        auto_start_on_startup: bool = False,
        replay_source: str = "offline_dataset_replay",
    ) -> dict[str, Any]:
        current = self.session_state() or {}
        payload = {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "started_at": _now_iso(),
            "status": "active",
            "metadata": metadata or {},
            "created_by": str((metadata or {}).get("created_by") or current.get("created_by") or "runtime_ui"),
            "persist_on_restart": bool(persist_on_restart),
            "auto_start_on_startup": bool(auto_start_on_startup),
            "replay_mode": str(replay_mode or "routing_override"),
            "replay_source": str(replay_source or "offline_dataset_replay"),
        }
        _write_json(self.active_session_path, payload)
        return payload

    def stop_session(self) -> dict[str, Any] | None:
        current = self.session_state()
        if current is None:
            return None
        current["status"] = "stopped"
        current["stopped_at"] = _now_iso()
        _write_json(self.active_session_path, current)
        return current

    def pause_session(self) -> dict[str, Any] | None:
        current = self.session_state()
        if current is None:
            return None
        current["status"] = "paused"
        current["paused_at"] = _now_iso()
        _write_json(self.active_session_path, current)
        return current

    def clear_session(self) -> bool:
        if not self.active_session_path.is_file():
            return False
        self.active_session_path.unlink(missing_ok=True)
        return True

    def session_state(self) -> dict[str, Any] | None:
        payload = _read_json(self.active_session_path)
        if not payload:
            return None
        dataset_id = str(payload.get("dataset_id") or "")
        session_id = str(payload.get("session_id") or "")
        if not dataset_id or not session_id:
            return None
        status = str(payload.get("status") or "inactive")
        exists = self.dataset_service.get_session(dataset_id, session_id) is not None
        return {
            **payload,
            "active": status == "active",
            "historical": status in {"stopped", "paused"},
            "dataset_exists": exists,
            "session_exists": exists,
        }

    def update_session_config(
        self,
        *,
        dataset_id: str | None = None,
        session_id: str | None = None,
        persist_on_restart: bool | None = None,
        auto_start_on_startup: bool | None = None,
        replay_mode: str | None = None,
        replay_source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.session_state() or {
            "dataset_id": "",
            "session_id": "",
            "status": "stopped",
            "metadata": {},
            "created_by": "runtime_ui",
            "started_at": None,
            "persist_on_restart": False,
            "auto_start_on_startup": False,
            "replay_mode": "routing_override",
            "replay_source": "offline_dataset_replay",
        }
        next_dataset = str(dataset_id if dataset_id is not None else current.get("dataset_id") or "")
        next_session = str(session_id if session_id is not None else current.get("session_id") or "")
        if next_dataset and next_session and self.dataset_service.get_session(next_dataset, next_session) is None:
            raise ValueError(f"Unknown dataset session: {next_dataset}/{next_session}")
        current["dataset_id"] = next_dataset
        current["session_id"] = next_session
        if persist_on_restart is not None:
            current["persist_on_restart"] = bool(persist_on_restart)
        if auto_start_on_startup is not None:
            current["auto_start_on_startup"] = bool(auto_start_on_startup)
        if replay_mode is not None:
            current["replay_mode"] = str(replay_mode)
        if replay_source is not None:
            current["replay_source"] = str(replay_source)
        if metadata is not None:
            current["metadata"] = metadata
        _write_json(self.active_session_path, current)
        return self.session_state() or current

    def active_session(self) -> dict[str, Any] | None:
        payload = self.session_state()
        if not payload:
            return None
        if str(payload.get("status") or "") != "active":
            return None
        if not bool(payload.get("dataset_exists")) or not bool(payload.get("session_exists")):
            return None
        return payload

    def resolve_session_for_take(self, *, source_metadata: dict[str, Any] | None = None) -> ReplaySessionRef | None:
        source_payload = source_metadata or {}
        dataset_id = str(source_payload.get("dataset_id") or "")
        session_id = str(source_payload.get("session_id") or "")
        if dataset_id and session_id and self.dataset_service.get_session(dataset_id, session_id):
            return ReplaySessionRef(dataset_id=dataset_id, session_id=session_id)
        active = self.active_session()
        if not active:
            return None
        return ReplaySessionRef(dataset_id=str(active["dataset_id"]), session_id=str(active["session_id"]))

    def attach_take_to_session(self, *, take_id: str, dataset_id: str, session_id: str, source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        updates = {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "acquisition_group_id": (source_metadata or {}).get("acquisition_group_id"),
        }
        return self.dataset_service.upsert_take_metadata(
            take_id=take_id,
            dataset_id=dataset_id,
            session_id=session_id,
            updates=updates,
            source_metadata=source_metadata,
        )

    def write_replay_manifest(
        self,
        *,
        take_dir: Path,
        take_id: str,
        source_id: str,
        metadata: dict[str, Any],
        parser_metadata: dict[str, Any] | None,
        session_ref: ReplaySessionRef | None,
        heightmap_metadata: dict[str, Any] | None = None,
        acquisition_metadata: dict[str, Any] | None = None,
        sync_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest_path = take_dir / "replay_manifest.json"
        existing = _read_json(manifest_path)
        if existing:
            return existing

        files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
        runtime = read_runtime_config().config
        calibration_file = runtime.default_calibration_file
        calibration_hash = None
        if calibration_file:
            calibration_path = Path(calibration_file)
            if calibration_path.is_file():
                calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

        payload: dict[str, Any] = {
            "take_id": take_id,
            "dataset_id": session_ref.dataset_id if session_ref else None,
            "session_id": session_ref.session_id if session_ref else None,
            "source_id": source_id,
            "timestamp": metadata.get("created_at") or _now_iso(),
            "modalities": metadata.get("modalities") or [],
            "assets": {
                "original_upload": files.get("raw_upload"),
                "heightmap": files.get("heightmap"),
                "reflectance": files.get("reflectance"),
                "heightmap_preview": files.get("heightmap_preview"),
                "parser_metadata": files.get("parser_metadata"),
            },
            "heightmap_metadata": heightmap_metadata or {
                "x_resolution_mm": None,
                "y_resolution_mm": None,
                "z_unit": "mm",
                "coordinate_system": "belt_relative",
            },
            "acquisition_metadata": acquisition_metadata or {},
            "calibration_snapshot": {
                "active_calibration_id": Path(calibration_file).stem if calibration_file else None,
                "default_calibration_file": calibration_file,
                "default_calibration_sha256": calibration_hash,
                "runtime_acquisition_config": runtime.model_dump(mode="json"),
            },
            "parser_metadata": parser_metadata or {},
            "acquisition_group_id": metadata.get("acquisition_group_id"),
            "frameset_id": ((metadata.get("frameset") or {}).get("frameset_id") if isinstance(metadata.get("frameset"), dict) else None),
            "sync_metadata": sync_metadata or {},
            "generated_at": _now_iso(),
            "immutable": True,
        }
        payload["manifest_sha256"] = _sha256_json(payload)
        _write_json(manifest_path, payload)
        return payload

    def replay_take(self, *, take_id: str, pipeline_id: str | None = None) -> dict[str, Any]:
        from vision_3d_acquisition.api.filesystem import read_json
        from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing
        from vision_3d_acquisition.api.settings import ApiSettings
        from vision_3d_acquisition.pipelines.registry import list_pipelines
        from vision_3d_acquisition.processes.bindings import ProcessBindingService

        settings = ApiSettings(
            data_dir=self.data_dir,
            incoming_dir=self.data_dir / "incoming",
            processed_dir=self.data_dir / "processed",
            state_dir=self.data_dir / "state",
            events_dir=self.data_dir / "events",
            sessions_dir=self.data_dir / "sessions",
            datasets_dir=self.data_dir / "datasets",
        )
        settings.ensure_directories()

        take_dir = settings.incoming_dir / take_id
        if not take_dir.is_dir():
            raise FileNotFoundError(f"Unknown take: {take_id}")
        manifest = read_json(take_dir / "replay_manifest.json") or {}
        modalities = [str(v) for v in (manifest.get("modalities") or [])]
        resolved_binding: dict[str, Any] | None = None
        selected_pipeline = pipeline_id
        if not selected_pipeline:
            modality = "heightmap" if ("heightmap" in modalities or "heightmap_2_5d" in modalities) else "rgb" if "rgb" in modalities else None
            if modality is None:
                raise ValueError("Could not resolve modality from manifest")
            source_id = str(manifest.get("source_id") or "")
            binding = ProcessBindingService(settings).resolve_process_binding(source_id, modality, "acquisition_inspection")
            resolved_binding = binding if isinstance(binding, dict) else None
            if resolved_binding is not None:
                selected_pipeline = str(resolved_binding.get("pipeline_id") or "")
            elif modality == "heightmap":
                selected_pipeline = "mining_steel_ball_classification_25d"
            else:
                selected_pipeline = "mining_steel_ball_classification_2d"

        valid_pipelines = {str(item.get("id") or "") for item in list_pipelines()}
        if selected_pipeline not in valid_pipelines:
            raise ValueError(f"Unknown pipeline: {selected_pipeline}")

        result = dispatch_take_processing(
            settings=settings,
            take_id=take_id,
            pipeline_id=selected_pipeline,
            reprocess=True,
            source_id=str(manifest.get("source_id") or ""),
            acquisition_group_id=str(manifest.get("acquisition_group_id") or "") or None,
            recipe_version_id=(str((resolved_binding or {}).get("active_recipe_version_id") or "") or None),
            calibration_profile_id=(str((resolved_binding or {}).get("calibration_profile_id") or "") or None),
        )
        return {
            "take_id": take_id,
            "pipeline_id": selected_pipeline,
            "result": result,
        }

    def replay_session(self, *, session_id: str, dataset_id: str | None = None, pipeline_id: str | None = None) -> dict[str, Any]:
        sessions = self.dataset_service.list_sessions(dataset_id=dataset_id)
        target = next((item for item in sessions if str(item.get("id") or "") == session_id), None)
        if target is None:
            raise FileNotFoundError(f"Unknown session: {session_id}")
        resolved_dataset_id = str(target.get("dataset_id") or "")
        takes_dir = self.data_dir / "datasets" / f"dataset_{resolved_dataset_id}" / "sessions" / f"session_{session_id}" / "takes"
        runs: list[dict[str, Any]] = []
        if takes_dir.is_dir():
            for child in sorted(takes_dir.iterdir()):
                if not child.is_dir():
                    continue
                take_id = child.name
                incoming_manifest = self.data_dir / "incoming" / take_id / "replay_manifest.json"
                if not incoming_manifest.is_file():
                    continue
                runs.append(self.replay_take(take_id=take_id, pipeline_id=pipeline_id))
        return {
            "dataset_id": resolved_dataset_id,
            "session_id": session_id,
            "replayed_takes": runs,
            "count": len(runs),
        }
