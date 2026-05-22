from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class HeightmapFrame:
    z_mm: np.ndarray
    valid_mask: np.ndarray
    reflectance: np.ndarray | None
    x_resolution_mm: float
    y_resolution_mm: float
    origin_x_mm: float
    origin_y_mm: float
    coordinate_system: str

    def pixel_area_mm2(self) -> float:
        return float(self.x_resolution_mm * self.y_resolution_mm)


def save_heightmap_npz(frame: HeightmapFrame, path: Path) -> None:
    np.savez_compressed(
        path,
        z_mm=np.asarray(frame.z_mm, dtype=np.float32),
        valid_mask=np.asarray(frame.valid_mask, dtype=np.uint8),
        reflectance=None if frame.reflectance is None else np.asarray(frame.reflectance),
        x_resolution_mm=np.float32(frame.x_resolution_mm),
        y_resolution_mm=np.float32(frame.y_resolution_mm),
        origin_x_mm=np.float32(frame.origin_x_mm),
        origin_y_mm=np.float32(frame.origin_y_mm),
        coordinate_system=np.array(frame.coordinate_system),
    )


def load_heightmap_npz(path: Path) -> HeightmapFrame:
    payload = np.load(path, allow_pickle=True)
    reflectance = payload.get("reflectance")
    if isinstance(reflectance, np.ndarray) and reflectance.shape == () and reflectance.item() is None:
        reflectance = None
    return HeightmapFrame(
        z_mm=np.asarray(payload["z_mm"], dtype=np.float32),
        valid_mask=np.asarray(payload["valid_mask"], dtype=bool),
        reflectance=None if reflectance is None else np.asarray(reflectance),
        x_resolution_mm=float(payload["x_resolution_mm"]),
        y_resolution_mm=float(payload["y_resolution_mm"]),
        origin_x_mm=float(payload["origin_x_mm"]),
        origin_y_mm=float(payload["origin_y_mm"]),
        coordinate_system=str(payload["coordinate_system"]),
    )


def write_heightmap_metadata_json(frame: HeightmapFrame, path: Path, *, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "shape": [int(frame.z_mm.shape[0]), int(frame.z_mm.shape[1])],
        "dtype": str(frame.z_mm.dtype),
        "valid_pixels": int(np.count_nonzero(frame.valid_mask)),
        "valid_ratio": float(np.count_nonzero(frame.valid_mask) / max(1, frame.valid_mask.size)),
        "x_resolution_mm": float(frame.x_resolution_mm),
        "y_resolution_mm": float(frame.y_resolution_mm),
        "origin_x_mm": float(frame.origin_x_mm),
        "origin_y_mm": float(frame.origin_y_mm),
        "coordinate_system": frame.coordinate_system,
        "has_reflectance": frame.reflectance is not None,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_heightmap_preview_png(values_mm: np.ndarray, valid_mask: np.ndarray, path: Path, *, colormap: int = cv2.COLORMAP_TURBO) -> None:
    values = np.asarray(values_mm, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    image = np.zeros(values.shape, dtype=np.uint8)
    if np.count_nonzero(mask) > 0:
        valid_values = values[mask]
        min_v = float(np.nanpercentile(valid_values, 2.0))
        max_v = float(np.nanpercentile(valid_values, 98.0))
        if max_v <= min_v:
            max_v = min_v + 1.0
        scaled = np.clip((values - min_v) / (max_v - min_v), 0.0, 1.0)
        image = np.asarray(np.round(scaled * 255.0), dtype=np.uint8)
        image[~mask] = 0
    color = cv2.applyColorMap(image, colormap)
    color[~mask] = (0, 0, 0)
    cv2.imwrite(str(path), color)
