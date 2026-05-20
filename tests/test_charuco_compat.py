from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vision_3d_acquisition.calibration import camera_2d


class _Board:
    def getDictionary(self):
        return object()


def test_detect_charuco_legacy_mode(monkeypatch) -> None:
    def interpolate(_mc, _mi, _gray, _board):
        return True, np.zeros((5, 1, 2), dtype=np.float32), np.arange(5, dtype=np.int32).reshape(-1, 1)

    aruco = SimpleNamespace(interpolateCornersCharuco=interpolate)
    monkeypatch.setattr(camera_2d, "_aruco_module", lambda: aruco)
    monkeypatch.setattr(camera_2d, "_detect_markers", lambda *_args, **_kwargs: ([np.zeros((4, 1, 2), dtype=np.float32)], np.array([[1], [2]], dtype=np.int32)))

    result = camera_2d.detect_charuco_corners_compat(np.zeros((32, 32), dtype=np.uint8), _Board())
    assert result["api_mode"] == "legacy"
    assert result["charuco_count"] == 5


def test_detect_charuco_detector_mode(monkeypatch) -> None:
    class Detector:
        def __init__(self, _board):
            pass

        def detectBoard(self, _gray):
            corners = np.zeros((6, 1, 2), dtype=np.float32)
            ids = np.arange(6, dtype=np.int32).reshape(-1, 1)
            return corners, ids, None, None

    aruco = SimpleNamespace(CharucoDetector=Detector)
    monkeypatch.setattr(camera_2d, "_aruco_module", lambda: aruco)
    monkeypatch.setattr(camera_2d, "_detect_markers", lambda *_args, **_kwargs: ([np.zeros((4, 1, 2), dtype=np.float32)], np.array([[1]], dtype=np.int32)))

    result = camera_2d.detect_charuco_corners_compat(np.zeros((32, 32), dtype=np.uint8), _Board())
    assert result["api_mode"] == "detector"
    assert result["charuco_count"] == 6


def test_detect_charuco_fallback_mode(monkeypatch) -> None:
    aruco = SimpleNamespace()
    monkeypatch.setattr(camera_2d, "_aruco_module", lambda: aruco)
    monkeypatch.setattr(camera_2d, "_detect_markers", lambda *_args, **_kwargs: ([np.zeros((4, 1, 2), dtype=np.float32)], np.array([[1], [2]], dtype=np.int32)))

    result = camera_2d.detect_charuco_corners_compat(np.zeros((32, 32), dtype=np.uint8), _Board())
    assert result["api_mode"] == "aruco_fallback"
    assert result["charuco_count"] == 0
    assert any("interpolation unavailable" in warning for warning in result["warnings"])


def test_aruco_diagnostics_missing_contrib(monkeypatch) -> None:
    fake_cv2 = SimpleNamespace(__version__="4.test", aruco=None)
    monkeypatch.setattr(camera_2d, "cv2", fake_cv2)

    result = camera_2d.aruco_runtime_diagnostics()
    assert result["cv2_version"] == "4.test"
    assert result["aruco_available"] is False
