from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest
from fastapi import HTTPException

from vision_3d_acquisition.api.calibration import (
    Capture2DFrameRequest,
    Detect2DCornersRequest,
    _acquire_capture_frame,
    capture_2d_image,
    capture_2d_frame,
    detect_2d_corners,
    list_2d_captures,
    _capture_record_path,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.calibration.calibration_models import Camera2DTargetConfig


def make_settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
    )
    settings.ensure_directories()
    return settings


def test_capture_works_even_when_preview_would_be_stale(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    called = {"acquire": 0}
    def fake_acquire(source_id: str, _settings: ApiSettings):
        called["acquire"] += 1
        assert source_id == "usb_camera_0"
        return {
            "frame": np.zeros((64, 96, 3), dtype=np.uint8),
            "acquisition_mode": "direct_source",
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 0.0,
        }

    monkeypatch.setattr("vision_3d_acquisition.api.calibration._acquire_capture_frame", fake_acquire)
    monkeypatch.setattr("vision_3d_acquisition.api.calibration._update_preview_from_capture", lambda *_args, **_kwargs: {"status": "live"})

    payload = capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)

    assert called["acquire"] == 1
    assert payload["source_id"] == "usb_camera_0"
    assert Path(payload["image_path"]).is_file()
    assert payload["acquisition_mode"] == "direct_source"


def test_detect_enabled_after_capture_and_captures_persist(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    monkeypatch.setattr(
        "vision_3d_acquisition.api.calibration._acquire_capture_frame",
        lambda *_args, **_kwargs: {
            "frame": np.zeros((40, 60, 3), dtype=np.uint8),
            "acquisition_mode": "preview_cache_fallback",
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 2.5,
        },
    )
    monkeypatch.setattr("vision_3d_acquisition.api.calibration._update_preview_from_capture", lambda *_args, **_kwargs: {"status": "live"})

    capture = capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)
    listed = list_2d_captures("usb_camera_0", settings)

    assert listed["captures"]
    assert listed["captures"][0]["id"] == capture["capture_id"]

    detect = detect_2d_corners(
        Detect2DCornersRequest(capture_ids=[capture["capture_id"]], target=Camera2DTargetConfig()),
        settings,
    )
    assert detect["capture_count"] == 1


def test_capture_uses_preview_cache_when_direct_unavailable(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    monkeypatch.setattr(
        "vision_3d_acquisition.api.calibration._acquire_capture_frame",
        lambda source_id, _settings: {
            "frame": np.zeros((40, 60, 3), dtype=np.uint8),
            "acquisition_mode": "preview_cache_fallback",
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 4.2,
            "freshness_status": "fresh",
            "fallback_used": True,
        },
    )
    monkeypatch.setattr("vision_3d_acquisition.api.calibration._update_preview_from_capture", lambda *_args, **_kwargs: {"status": "live"})
    payload = capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)
    assert payload["source_id"] == "usb_camera_0"
    assert payload["acquisition_mode"] == "preview_cache_fallback"
    assert payload["frame_age_seconds"] == 4.2
    assert payload["freshness_status"] == "fresh"
    assert payload["fallback_used"] is True


def test_source_id_maps_to_usb_camera_index_and_uses_reusable_capture(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    called = {"index": None}

    class Direct:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        backend = "AVFOUNDATION"
        timestamp = "2026-01-01T00:00:00+00:00"

    def fake_direct(*, data_dir: Path, camera_index: int, session_id: str | None):
        assert data_dir == settings.data_dir
        called["index"] = camera_index
        assert session_id is None
        return Direct()

    monkeypatch.setattr("vision_3d_acquisition.api.calibration.capture_single_frame", fake_direct)
    result = _acquire_capture_frame("usb_camera_0", settings)
    assert called["index"] == 0
    assert result["acquisition_mode"] == "direct_usb_camera"
    assert result["direct_capture_success"] is True


def test_fallback_only_after_direct_failure(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fail_direct(*_args, **_kwargs):
        raise RuntimeError("camera busy")

    monkeypatch.setattr("vision_3d_acquisition.api.calibration.capture_single_frame", fail_direct)
    monkeypatch.setattr(
        "vision_3d_acquisition.api.calibration._refresh_preview_cache",
        lambda *_args, **_kwargs: {
            "frame": np.zeros((20, 30, 3), dtype=np.uint8),
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 0.3,
        },
    )
    result = _acquire_capture_frame("usb_camera_0", settings)
    assert result["acquisition_mode"] == "preview_cache_fallback"
    assert result["fallback_used"] is True
    assert result["direct_capture_success"] is False


def test_old_preview_cache_does_not_create_capture(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    monkeypatch.setattr(
        "vision_3d_acquisition.api.calibration._acquire_capture_frame",
        lambda *_args, **_kwargs: {
            "frame": np.zeros((40, 60, 3), dtype=np.uint8),
            "acquisition_mode": "preview_cache_fallback",
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 9.9,
            "freshness_status": "stale",
            "fallback_used": True,
        },
    )
    with pytest.raises(HTTPException) as exc:
        capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "NO_FRESH_FRAME_AVAILABLE"


def test_capture_returns_structured_error(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fail(*_args, **_kwargs):
        raise HTTPException(status_code=404, detail={"code": "NO_PREVIEW_FRAME_AVAILABLE", "message": "No preview frame available"})

    monkeypatch.setattr("vision_3d_acquisition.api.calibration._acquire_capture_frame", fail)
    with pytest.raises(HTTPException) as exc:
        capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)
    assert isinstance(exc.value.detail, dict)
    assert exc.value.detail["code"] == "NO_PREVIEW_FRAME_AVAILABLE"


def test_capture_image_route_returns_file(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    monkeypatch.setattr(
        "vision_3d_acquisition.api.calibration._acquire_capture_frame",
        lambda *_args, **_kwargs: {
            "frame": np.zeros((32, 48, 3), dtype=np.uint8),
            "acquisition_mode": "preview_cache",
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 0.0,
        },
    )
    monkeypatch.setattr("vision_3d_acquisition.api.calibration._update_preview_from_capture", lambda *_args, **_kwargs: {"status": "live"})
    payload = capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)
    response = capture_2d_image(payload["capture_id"], settings)
    assert response.path.endswith(".png")


def test_capture_image_route_rejects_invalid_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    with pytest.raises(HTTPException) as exc:
        capture_2d_image("missing_capture", settings)
    assert exc.value.status_code == 404


def test_capture_image_route_rejects_path_outside_captures_dir(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    monkeypatch.setattr(
        "vision_3d_acquisition.api.calibration._acquire_capture_frame",
        lambda *_args, **_kwargs: {
            "frame": np.zeros((32, 48, 3), dtype=np.uint8),
            "acquisition_mode": "preview_cache",
            "frame_timestamp": "2026-01-01T00:00:00+00:00",
            "frame_age_seconds": 0.0,
        },
    )
    monkeypatch.setattr("vision_3d_acquisition.api.calibration._update_preview_from_capture", lambda *_args, **_kwargs: {"status": "live"})
    payload = capture_2d_frame(Capture2DFrameRequest(source_id="usb_camera_0"), settings)
    record_path = _capture_record_path(settings, payload["capture_id"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["image_path"] = "/tmp/evil.png"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        capture_2d_image(payload["capture_id"], settings)
    assert exc.value.status_code == 403
