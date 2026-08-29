from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.main import CaptureTakeRequest, UsbCaptureRequest, capture_rgb_image, capture_take_to_dataset, takes
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


def _fake_capture_image(*, data_dir: Path, camera_index: int, session_id: str | None = None, **kwargs):
    take_id = "captured_take_001"
    folder = data_dir / "incoming" / take_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rgb.png").write_bytes(b"fake")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "created_at": "2026-05-20T12:00:00Z",
                "session_id": session_id,
                "files": {"rgb": "rgb.png"},
            }
        ),
        encoding="utf-8",
    )
    (folder / "READY").touch()
    return take_id, folder


def test_capture_to_dataset_creates_raw_take_and_metadata(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    monkeypatch.setattr("vision_3d_acquisition.api.main.capture_image", _fake_capture_image)

    resp = capture_take_to_dataset(
        CaptureTakeRequest(
            dataset_id="d1",
            dataset_session_id="s1",
            friendly_name="POC Take",
            tags=["good", "wet"],
            expected_class="ball",
            expected_diameter_mm=80,
            notes="captured in studio",
        ),
        settings,
    )

    take_id = resp["take_id"]
    assert (settings.incoming_dir / take_id / "metadata.json").is_file()
    detail = resp["take"]
    assert detail is not None
    assert detail["take_metadata"]["dataset_id"] == "d1"
    assert detail["take_metadata"]["session_id"] == "s1"
    assert detail["take_metadata"]["friendly_name"] == "POC Take"
    assert detail["take_metadata"]["validation_status"] == "unreviewed"


def test_captured_take_appears_in_dataset_filter(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    monkeypatch.setattr("vision_3d_acquisition.api.main.capture_image", _fake_capture_image)

    capture_take_to_dataset(CaptureTakeRequest(dataset_id="d1", dataset_session_id="s1"), settings)

    filtered = takes(None, settings, dataset_id="d1")

    assert len(filtered) == 1
    assert filtered[0].dataset_id == "d1"


def test_old_capture_endpoint_still_works(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    monkeypatch.setattr("vision_3d_acquisition.api.main.capture_image", _fake_capture_image)

    resp = capture_rgb_image(UsbCaptureRequest(camera_index=0), settings)

    assert resp["take_id"] == "captured_take_001"
    assert resp["acquisition_processing"]["status"] == "processing_binding_missing"
    assert "No active processing binding found" in str(resp["acquisition_processing"].get("warning") or "")
    assert (settings.incoming_dir / "captured_take_001" / "metadata.json").is_file()
