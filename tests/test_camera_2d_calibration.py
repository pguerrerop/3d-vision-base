from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_3d_acquisition.calibration.calibration_models import (
    Camera2DBeltPlane,
    Camera2DIntrinsics,
    Camera2DTargetConfig,
    SystemCalibration,
)
from vision_3d_acquisition.calibration.calibration_storage import load_calibration, save_calibration
from vision_3d_acquisition.calibration.camera_2d import axis_length_mm, pixel_to_mm
from vision_3d_acquisition.calibration.charuco import generate_charuco_board, save_charuco_pdf, save_charuco_png


def test_camera_2d_schema_serialization_roundtrip(tmp_path: Path) -> None:
    calibration = SystemCalibration(
        calibration_id="camera2d_test",
        calibration_type="camera_2d",
        source_modalities=["rgb"],
        source_id="usb_camera_0",
        target=Camera2DTargetConfig(),
        intrinsics=Camera2DIntrinsics(
            camera_matrix=[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            dist_coeffs=[0.1, -0.01, 0.0, 0.0, 0.0],
            reprojection_error=0.2,
            image_width=1280,
            image_height=720,
        ),
        belt_plane=Camera2DBeltPlane(
            homography=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            mm_per_px_x=0.5,
            mm_per_px_y=0.5,
            reference_plane_z_mm=0.0,
        ),
    )

    path = save_calibration(calibration, tmp_path)
    loaded = load_calibration(path, tmp_path)

    assert loaded.calibration_type == "camera_2d"
    assert loaded.source_id == "usb_camera_0"
    assert loaded.belt_plane is not None
    assert loaded.belt_plane.mm_per_px_x == 0.5


def test_pixel_to_mm_and_axis_length_conversion() -> None:
    h = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
    mapped = pixel_to_mm((20.0, 10.0), h)
    assert mapped == (10.0, 5.0)

    major_mm = axis_length_mm((50.0, 50.0), 20.0, 0.0, h)
    assert major_mm == pytest.approx(20.0, abs=1e-6)


def test_charuco_generation_and_exports(tmp_path: Path) -> None:
    try:
        board = generate_charuco_board(
            squares_x=7,
            squares_y=5,
            square_length_mm=25.0,
            marker_length_mm=18.0,
            dictionary="DICT_4X4_50",
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert isinstance(board, np.ndarray)
    png = save_charuco_png(tmp_path / "charuco.png", board)
    pdf = save_charuco_pdf(tmp_path / "charuco.pdf", board)
    assert png.is_file()
    assert pdf.is_file()


def test_target_dictionary_and_dimensions_validation() -> None:
    cfg = Camera2DTargetConfig(dictionary="DICT_6X6_250", square_length_mm=30.0, marker_length_mm=20.0)
    assert cfg.dictionary == "DICT_6X6_250"
    with pytest.raises(ValueError):
        Camera2DTargetConfig(dictionary="DICT_7X7_50")
    with pytest.raises(ValueError):
        Camera2DTargetConfig(square_length_mm=20.0, marker_length_mm=20.0)
