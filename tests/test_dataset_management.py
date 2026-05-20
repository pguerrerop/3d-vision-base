from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.filesystem import get_take_detail, list_takes
from vision_3d_acquisition.api.main import update_take_metadata
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


def make_settings(data_dir: Path) -> ApiSettings:
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


def write_take(data_dir: Path, take_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-19T10:00:00Z", "session_id": "session_live", "files": {"rgb": "rgb.png"}}),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def test_take_metadata_defaults_for_legacy_take(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_legacy")

    detail = get_take_detail(settings, "take_legacy")

    assert detail is not None
    assert detail.take_metadata["friendly_name"] == "take_legacy"
    assert detail.take_metadata["validation_status"] == "unreviewed"


def test_take_metadata_persistence_and_tag_editing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_meta")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="s1", name="Session 1")
    service.upsert_take_metadata(
        take_id="take_meta",
        dataset_id="demo",
        session_id="s1",
        updates={"friendly_name": "Take Friendly", "tags": ["good", "wet"], "expected_diameter_mm": 42.5},
        source_metadata={"session_id": "session_live"},
    )

    detail = get_take_detail(settings, "take_meta")
    summary = list_takes(settings)[0]

    assert detail is not None
    assert detail.take_metadata["friendly_name"] == "Take Friendly"
    assert set(detail.take_metadata["tags"]) == {"good", "wet"}
    assert summary.expected_diameter_mm == 42.5


def test_dataset_session_grouping_and_take_filter(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_a")
    write_take(settings.data_dir, "take_b")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="daylight", name="Daylight")
    service.upsert_take_metadata(take_id="take_a", dataset_id="set1", session_id="daylight", updates={"tags": ["good"]}, source_metadata={})

    all_takes = list_takes(settings)
    grouped = list_takes(settings, dataset_id="set1")
    tagged = list_takes(settings, tag="good")

    assert len(all_takes) == 2
    assert [item.take_id for item in grouped] == ["take_a"]
    assert [item.take_id for item in tagged] == ["take_a"]


def test_run_history_lookup_includes_process_entries(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_hist")
    index_dir = settings.data_dir / "processes" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "runs.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "take_id": "take_hist",
                        "pipeline_instance_id": "pipe_1",
                        "run_id": "run_1",
                        "pipeline_family": "2d",
                        "status": "success",
                        "created_at": "2026-05-19T12:00:00Z",
                        "path": "data/processes/runs/pipe_1/run_1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    detail = get_take_detail(settings, "take_hist")

    assert detail is not None
    assert detail.run_history
    assert detail.run_history[0]["run_id"] == "run_1"
