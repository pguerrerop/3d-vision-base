from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.calibration.calibration_models import Camera2DTargetConfig


@dataclass
class DetectedFrame:
    image_path: str
    image_width: int
    image_height: int
    corners_found: bool
    corner_count: int
    image_points: np.ndarray | None
    object_points: np.ndarray | None
    debug_image: np.ndarray
    warnings: list[str]
    marker_count: int = 0
    api_mode: str = "legacy"
    success: bool = False
    error_code: str | None = None


def _aruco_module() -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco module not available. Install opencv-contrib-python.")
    return aruco


def _charuco_board(target: Camera2DTargetConfig) -> Any:
    aruco = _aruco_module()
    dictionary_id = getattr(aruco, target.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {target.dictionary}")
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard((int(target.squares_x), int(target.squares_y)), float(target.square_length_mm), float(target.marker_length_mm), dictionary)
    creator = getattr(aruco, "CharucoBoard_create", None)
    if callable(creator):
        return creator(int(target.squares_x), int(target.squares_y), float(target.square_length_mm), float(target.marker_length_mm), dictionary)
    raise RuntimeError("ChArUco board creation unavailable in this OpenCV build.")


def aruco_runtime_diagnostics() -> dict[str, Any]:
    aruco = getattr(cv2, "aruco", None)
    attrs = ("ArucoDetector", "CharucoDetector", "CharucoBoard", "CharucoBoard_create", "detectMarkers", "interpolateCornersCharuco")
    return {
        "cv2_version": str(getattr(cv2, "__version__", "unknown")),
        "aruco_available": aruco is not None,
        "aruco_attrs": {name: bool(hasattr(aruco, name)) if aruco is not None else False for name in attrs},
    }


def _detect_markers(gray: np.ndarray, board: Any, aruco: Any) -> tuple[list[Any], Any]:
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(board.getDictionary())
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        return marker_corners, marker_ids
    detect = getattr(aruco, "detectMarkers", None)
    if callable(detect):
        marker_corners, marker_ids, _ = detect(gray, board.getDictionary())
        return marker_corners, marker_ids
    return [], None


def detect_charuco_corners_compat(gray: np.ndarray, board: Any) -> dict[str, Any]:
    aruco = _aruco_module()
    warnings: list[str] = []
    marker_corners, marker_ids = _detect_markers(gray, board, aruco)
    marker_count = int(len(marker_ids)) if marker_ids is not None else 0

    if hasattr(aruco, "CharucoDetector"):
        try:
            detector = aruco.CharucoDetector(board)
            out = detector.detectBoard(gray)
            if isinstance(out, tuple) and len(out) >= 2:
                charuco_corners, charuco_ids = out[0], out[1]
            else:
                charuco_corners, charuco_ids = None, None
            count = int(len(charuco_ids)) if charuco_ids is not None else 0
            return {
                "api_mode": "detector",
                "marker_corners": marker_corners,
                "marker_ids": marker_ids,
                "marker_count": marker_count,
                "charuco_corners": charuco_corners,
                "charuco_ids": charuco_ids,
                "charuco_count": count,
                "warnings": warnings,
            }
        except Exception:
            warnings.append("CharucoDetector path failed; trying legacy interpolation.")

    if hasattr(aruco, "interpolateCornersCharuco"):
        success, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
        count = int(len(charuco_ids)) if charuco_ids is not None else 0
        return {
            "api_mode": "legacy",
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "marker_count": marker_count,
            "charuco_corners": charuco_corners if success else None,
            "charuco_ids": charuco_ids if success else None,
            "charuco_count": count if success else 0,
            "warnings": warnings,
        }

    warnings.append("ChArUco interpolation unavailable; using ArUco marker corners only.")
    return {
        "api_mode": "aruco_fallback",
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "marker_count": marker_count,
        "charuco_corners": None,
        "charuco_ids": None,
        "charuco_count": 0,
        "warnings": warnings,
    }


def detect_target_in_image(image_path: Path, target: Camera2DTargetConfig) -> DetectedFrame:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or invalid: {image_path}")
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug = image.copy()
    warnings: list[str] = []

    if target.type == "charuco":
        aruco = _aruco_module()
        board = _charuco_board(target)
        compat = detect_charuco_corners_compat(gray, board)
        marker_corners = compat["marker_corners"]
        marker_ids = compat["marker_ids"]
        marker_count = int(compat.get("marker_count", 0))
        warnings.extend(list(compat.get("warnings", [])))
        api_mode = str(compat.get("api_mode", "legacy"))
        if marker_ids is None or len(marker_ids) == 0:
            return DetectedFrame(str(image_path), w, h, False, 0, None, None, debug, warnings, marker_count=0, api_mode=api_mode, success=False, error_code="NO_CORNERS_DETECTED")
        aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
        charuco_corners = compat.get("charuco_corners")
        charuco_ids = compat.get("charuco_ids")
        charuco_count = int(compat.get("charuco_count", 0))
        if charuco_ids is None or charuco_corners is None or charuco_count < 4:
            code = "CHARUCO_API_UNAVAILABLE" if api_mode == "aruco_fallback" else "NO_CORNERS_DETECTED"
            return DetectedFrame(str(image_path), w, h, False, charuco_count, None, None, debug, warnings, marker_count=marker_count, api_mode=api_mode, success=False, error_code=code)
        aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
        obj_points = board.getChessboardCorners()[charuco_ids.flatten()].astype(np.float32)
        img_points = charuco_corners.reshape(-1, 2).astype(np.float32)
        return DetectedFrame(str(image_path), w, h, True, int(len(charuco_ids)), img_points, obj_points, debug, warnings, marker_count=marker_count, api_mode=api_mode, success=True, error_code=None)

    board_size = (int(target.squares_x), int(target.squares_y))
    ok, corners = cv2.findChessboardCorners(gray, board_size)
    if not ok or corners is None:
        return DetectedFrame(str(image_path), w, h, False, 0, None, None, debug, warnings, marker_count=0, api_mode="checkerboard", success=False, error_code="NO_CORNERS_DETECTED")
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    cv2.drawChessboardCorners(debug, board_size, refined, True)
    obj = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    obj[:, :2] = obj[:, :2] * float(target.square_length_mm)
    return DetectedFrame(str(image_path), w, h, True, int(refined.shape[0]), refined.reshape(-1, 2).astype(np.float32), obj, debug, warnings, marker_count=0, api_mode="checkerboard", success=True, error_code=None)


def calibrate_from_detections(frames: list[DetectedFrame]) -> dict[str, Any]:
    usable = [frame for frame in frames if frame.corners_found and frame.image_points is not None and frame.object_points is not None]
    if not usable:
        raise ValueError("No valid corner detections available")
    img_size = (int(usable[0].image_width), int(usable[0].image_height))
    object_points = [frame.object_points.reshape(-1, 1, 3).astype(np.float32) for frame in usable]
    image_points = [frame.image_points.reshape(-1, 1, 2).astype(np.float32) for frame in usable]
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)

    reprojection_errors: list[float] = []
    residual_rows: list[dict[str, float | int]] = []
    for frame_idx, (obj, img, rvec, tvec) in enumerate(zip(object_points, image_points, rvecs, tvecs, strict=False)):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        proj2 = projected.reshape(-1, 2)
        img2 = img.reshape(-1, 2)
        deltas = np.linalg.norm(proj2 - img2, axis=1)
        reprojection_errors.extend([float(value) for value in deltas])
        for corner_idx, value in enumerate(deltas):
            residual_rows.append({"frame_index": frame_idx, "corner_index": corner_idx, "error_px": float(value)})

    reproj_mean = float(np.mean(reprojection_errors)) if reprojection_errors else float(rms)

    homography, mask = cv2.findHomography(image_points[0].reshape(-1, 2), object_points[0].reshape(-1, 3)[:, :2], method=cv2.RANSAC)
    if homography is None:
        raise ValueError("Failed to estimate belt-plane homography")

    mm_x, mm_y = estimate_mm_per_px(homography, img_size)

    return {
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).astype(float).tolist(),
        "reprojection_error": reproj_mean,
        "image_width": img_size[0],
        "image_height": img_size[1],
        "homography": homography.astype(float).tolist(),
        "mm_per_px_x": mm_x,
        "mm_per_px_y": mm_y,
        "used_frames": len(usable),
        "residuals": residual_rows,
        "inlier_count": int(np.sum(mask)) if mask is not None else len(image_points[0]),
    }


def estimate_mm_per_px(homography: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float]:
    width, height = int(image_size[0]), int(image_size[1])
    p0 = pixel_to_mm((width / 2.0, height / 2.0), homography)
    px = pixel_to_mm((width / 2.0 + 1.0, height / 2.0), homography)
    py = pixel_to_mm((width / 2.0, height / 2.0 + 1.0), homography)
    mm_per_px_x = float(np.linalg.norm(np.array(px) - np.array(p0)))
    mm_per_px_y = float(np.linalg.norm(np.array(py) - np.array(p0)))
    return max(mm_per_px_x, 1e-9), max(mm_per_px_y, 1e-9)


def pixel_to_mm(pixel_xy: tuple[float, float], homography: np.ndarray | list[list[float]]) -> tuple[float, float]:
    h = np.asarray(homography, dtype=float)
    vec = np.array([float(pixel_xy[0]), float(pixel_xy[1]), 1.0], dtype=float)
    mapped = h @ vec
    if abs(mapped[2]) < 1e-12:
        return (float("nan"), float("nan"))
    return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))


def undistort_image(image: np.ndarray, camera_matrix: list[list[float]], dist_coeffs: list[float]) -> np.ndarray:
    cm = np.asarray(camera_matrix, dtype=float)
    dc = np.asarray(dist_coeffs, dtype=float)
    return cv2.undistort(image, cm, dc)


def axis_length_mm(center_xy: tuple[float, float], radius_px: float, angle_rad: float, homography: list[list[float]]) -> float:
    p1 = (center_xy[0] - radius_px * np.cos(angle_rad), center_xy[1] - radius_px * np.sin(angle_rad))
    p2 = (center_xy[0] + radius_px * np.cos(angle_rad), center_xy[1] + radius_px * np.sin(angle_rad))
    m1 = pixel_to_mm(p1, homography)
    m2 = pixel_to_mm(p2, homography)
    return float(np.linalg.norm(np.asarray(m2) - np.asarray(m1)))
