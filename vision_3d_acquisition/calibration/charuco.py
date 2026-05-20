from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _aruco_module() -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco module not available. Install opencv-contrib-python.")
    return aruco


def _dictionary_by_name(dictionary: str) -> Any:
    aruco = _aruco_module()
    key = getattr(aruco, dictionary, None)
    if key is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary}")
    return aruco.getPredefinedDictionary(key)


def generate_charuco_board(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary: str = "DICT_4X4_50",
    margin_px: int = 24,
    px_per_mm: float = 6.0,
) -> np.ndarray:
    aruco = _aruco_module()
    board = aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_length_mm), float(marker_length_mm), _dictionary_by_name(dictionary))
    width_px = max(32, int(round(squares_x * square_length_mm * px_per_mm)))
    height_px = max(32, int(round(squares_y * square_length_mm * px_per_mm)))
    image = board.generateImage((width_px, height_px), marginSize=max(0, int(margin_px)), borderBits=1)
    return image


def save_charuco_png(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to save ChArUco PNG: {path}")
    return path


def save_charuco_pdf(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(image)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil.save(path, "PDF", resolution=300.0)
    return path
