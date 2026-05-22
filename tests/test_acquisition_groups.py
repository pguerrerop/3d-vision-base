from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.api.main import (
    CaptureTakeRequest,
    CreateAcquisitionGroupRequest,
    ExecuteTakeRequest,
    UsbCaptureRequest,
    attach_take_acquisition_group,
    capture_rgb_image,
    capture_take_to_dataset,
    create_acquisition_group,
    get_acquisition_group,
    list_acquisition_group_takes,
    process_take_for_pipeline,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.processes.service import ProcessService


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


def _fake_capture_image(*, data_dir: Path, camera_index: int, session_id: str | None = None, acquisition_group_id: str | None = None, **kwargs):
    take_id = "captured_take_group_001"
    folder = data_dir / "incoming" / take_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "take_id": take_id,
        "created_at": "2026-05-21T12:00:00Z",
        "session_id": session_id,
        "files": {"rgb": "rgb.png"},
    }
    if acquisition_group_id:
        payload["acquisition_group_id"] = acquisition_group_id
    (folder / "rgb.png").write_bytes(b"fake")
    (folder / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    (folder / "READY").touch()
    return take_id, folder


def _write_rgb_take(settings: ApiSettings, take_id: str, *, acquisition_group_id: str | None = None) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(img, (60, 50), 24, (220, 220, 220), -1)
    cv2.imwrite(str(take_dir / "rgb.png"), img)
    meta = {
        "take_id": take_id,
        "source": "usb_camera_0",
        "created_at": "2026-05-21T12:00:00Z",
        "files": {"rgb": "rgb.png"},
        "modalities": ["rgb"],
    }
    if acquisition_group_id:
        meta["acquisition_group_id"] = acquisition_group_id
    (take_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (take_dir / "READY").touch()


def _prepare_process_recipe(settings: ApiSettings) -> str:
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB POC",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    recipe = service.promote_recipe(instance["id"], run_id=None, recipe_type="production")
    return recipe["id"]


def test_create_group(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    payload = create_acquisition_group(
        CreateAcquisitionGroupRequest(name="Line A", station_id="st01", trigger_id="trg-1", encoder_position=12.5),
        settings,
    )
    assert payload["id"].startswith("ag_")
    assert payload["status"] == "open"


def test_attach_take_to_group_and_list_takes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="manual")
    _write_rgb_take(settings, "take_group_01")

    attached = attach_take_acquisition_group(group["id"], "take_group_01", settings)
    assert attached["ok"] is True

    takes = list_acquisition_group_takes(group["id"], settings)
    assert len(takes) == 1
    assert takes[0]["take_id"] == "take_group_01"


def test_reject_missing_or_invalid_group(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    monkeypatch.setattr("vision_3d_acquisition.api.main.capture_image", _fake_capture_image)
    with pytest.raises(Exception):
        capture_rgb_image(UsbCaptureRequest(camera_index=0, acquisition_group_id="ag_missing"), settings)


def test_process_run_includes_acquisition_group_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="event-1")
    _write_rgb_take(settings, "take_group_proc", acquisition_group_id=group["id"])
    _prepare_process_recipe(settings)

    response = process_take_for_pipeline(
        "take_group_proc",
        ExecuteTakeRequest(pipeline_id="mining_steel_ball_classification_2d", source_id="usb_camera_0", modality="rgb"),
        settings,
    )
    assert response["ok"] is True
    index = json.loads((settings.data_dir / "processes" / "index" / "runs.json").read_text(encoding="utf-8"))
    latest = index["entries"][-1]
    assert latest["acquisition_group_id"] == group["id"]


def test_old_captures_still_work_without_group(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    monkeypatch.setattr("vision_3d_acquisition.api.main.capture_image", _fake_capture_image)
    resp = capture_rgb_image(UsbCaptureRequest(camera_index=0), settings)
    assert resp["take_id"] == "captured_take_group_001"


def test_capture_take_to_dataset_persists_group_metadata(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="event-2")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    monkeypatch.setattr("vision_3d_acquisition.api.main.capture_image", _fake_capture_image)

    resp = capture_take_to_dataset(
        CaptureTakeRequest(dataset_id="d1", dataset_session_id="s1", acquisition_group_id=group["id"]),
        settings,
    )
    assert resp["take"]["take_metadata"]["acquisition_group_id"] == group["id"]


def test_get_group_endpoint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="group-x")
    loaded = get_acquisition_group(group["id"], settings)
    assert loaded["id"] == group["id"]
