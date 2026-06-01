from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.dataset_session_exports import DatasetSessionExportService
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


def _settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def _write_take(data_dir: Path, take_id: str, live_session_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "created_at": "2026-05-31T12:00:00Z",
                "session_id": live_session_id,
                "files": {"rgb": "rgb.png"},
            }
        ),
        encoding="utf-8",
    )


def test_session_export_and_summary_use_canonical_dataset_session_membership(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="ds1", name="Dataset 1")
    service.create_session(dataset_id="ds1", session_id="s_day", name="Day")
    service.create_session(dataset_id="ds1", session_id="s_night", name="Night")

    for take_id in ("a1", "a2", "a3"):
        _write_take(settings.data_dir, take_id, live_session_id="runtime_shared")
        service.upsert_take_metadata(
            take_id=take_id,
            dataset_id="ds1",
            session_id="s_day",
            updates={"validation_status": "valid"},
            source_metadata={"session_id": "runtime_shared"},
        )
    for take_id in ("b1", "b2"):
        _write_take(settings.data_dir, take_id, live_session_id="runtime_shared")
        service.upsert_take_metadata(
            take_id=take_id,
            dataset_id="ds1",
            session_id="s_night",
            updates={"validation_status": "unreviewed"},
            source_metadata={"session_id": "runtime_shared"},
        )

    export_service = DatasetSessionExportService(settings)
    day_summary = export_service.build_session_summary("s_day", "ds1")
    day_export = export_service.export_session("s_day", "ds1")
    night_summary = export_service.build_session_summary("s_night", "ds1")

    assert day_summary["summary"]["total_takes"] == 3
    assert len(day_export["takes"]) == 3
    assert all(row["metadata"].get("session_id") == "s_day" for row in day_export["takes"])

    assert night_summary["summary"]["total_takes"] == 2
    assert day_summary["summary"]["total_takes"] != night_summary["summary"]["total_takes"]
