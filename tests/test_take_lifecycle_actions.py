from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from vision_3d_acquisition.api.filesystem import list_takes
from vision_3d_acquisition.api.main import (
    ArchiveTakeRequest,
    PermanentDeleteTakeRequest,
    archive_take,
    permanent_delete_take,
    remove_take_from_dataset,
    restore_take,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.processing.status_index import append_process_run_index
from vision_3d_acquisition.storage.run_store import runs_for_take


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


def write_take(settings: ApiSettings, take_id: str) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(json.dumps({"take_id": take_id, "created_at": "2026-05-20T12:00:00Z", "files": {"rgb": "rgb.png"}}), encoding="utf-8")
    (take_dir / "READY").touch()


def attach_to_dataset(settings: ApiSettings, take_id: str) -> None:
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={})


def test_remove_from_dataset_preserves_raw_take(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_remove")
    attach_to_dataset(settings, "take_remove")

    remove_take_from_dataset("take_remove", settings)

    assert (settings.incoming_dir / "take_remove" / "metadata.json").is_file()
    filtered = list_takes(settings, dataset_id="d1")
    assert not filtered


def test_archive_hidden_by_default_and_restore(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_archive")
    attach_to_dataset(settings, "take_archive")

    archive_take("take_archive", ArchiveTakeRequest(reason="obsolete"), settings)

    default_list = list_takes(settings)
    archived_list = list_takes(settings, include_archived=True)
    assert not any(item.take_id == "take_archive" for item in default_list)
    assert any(item.take_id == "take_archive" for item in archived_list)

    restore_take("take_archive", settings)
    restored = list_takes(settings)
    assert any(item.take_id == "take_archive" for item in restored)


def test_permanent_delete_removes_process_run_rows(tmp_path: Path) -> None:
    """delete_runs used to edit the runs.json mirror, which a full reindex

    regenerates from process_run — so a "permanently deleted" take's runs
    silently came back, or rather never actually left the database. This
    exercises the real path: a run recorded through append_process_run_index,
    which is how every worker actually writes one.
    """
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_with_runs")
    attach_to_dataset(settings, "take_with_runs")
    run_dir = settings.data_dir / "processes" / "runs" / "pipeline_1" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    append_process_run_index(
        settings.data_dir,
        take_id="take_with_runs",
        pipeline_instance_id="pipeline_1",
        run_id="run_1",
        pipeline_family="25d",
        status="success",
        run_dir=run_dir,
        created_at="2026-05-20T12:05:00Z",
    )
    assert runs_for_take(settings.data_dir, "take_with_runs")

    permanent_delete_take(
        "take_with_runs", PermanentDeleteTakeRequest(confirmation="take_with_runs"), settings
    )

    assert runs_for_take(settings.data_dir, "take_with_runs") == []
    assert not run_dir.exists()


def test_permanent_delete_requires_confirmation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_delete")
    attach_to_dataset(settings, "take_delete")

    with pytest.raises(HTTPException):
        permanent_delete_take("take_delete", PermanentDeleteTakeRequest(confirmation="wrong"), settings)

    permanent_delete_take("take_delete", PermanentDeleteTakeRequest(confirmation="take_delete"), settings)
    assert not (settings.incoming_dir / "take_delete").exists()
