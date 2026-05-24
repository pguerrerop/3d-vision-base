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


def write_heightmap_preview_png(
    values_mm: np.ndarray,
    valid_mask: np.ndarray,
    path: Path,
    *,
    colormap: int = cv2.COLORMAP_TURBO,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    clipping_percentiles: tuple[float, float] = (2.0, 98.0),
) -> None:
    values = np.asarray(values_mm, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    image = np.zeros(values.shape, dtype=np.uint8)
    min_v = 0.0
    max_v = 1.0
    if np.count_nonzero(mask) > 0:
        valid_values = values[mask]
        if vmin is None or vmax is None:
            p_lo, p_hi = clipping_percentiles
            min_v = float(np.nanpercentile(valid_values, p_lo))
            max_v = float(np.nanpercentile(valid_values, p_hi))
        else:
            min_v = float(vmin)
            max_v = float(vmax)
        if max_v <= min_v:
            max_v = min_v + 1.0
        scaled = np.clip((values - min_v) / (max_v - min_v), 0.0, 1.0)
        image = np.asarray(np.round(scaled * 255.0), dtype=np.uint8)
        image[~mask] = 0
    color = cv2.applyColorMap(image, colormap)
    color[~mask] = (0, 0, 0)
    if colorbar_label:
        color = _append_colorbar(color, min_v=min_v, max_v=max_v, label=colorbar_label, colormap=colormap)
    cv2.imwrite(str(path), color)


def _append_colorbar(image: np.ndarray, *, min_v: float, max_v: float, label: str, colormap: int) -> np.ndarray:
    h, w = image.shape[:2]
    bar_h = 46
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h, :, :] = image
    canvas[h:, :, :] = (18, 18, 18)
    margin = max(24, min(80, w // 12))
    bar_w = max(1, w - (2 * margin))
    gradient = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, bar_w)
    gradient = np.repeat(gradient, 12, axis=0)
    bar = cv2.applyColorMap(gradient, colormap)
    y0 = h + 8
    canvas[y0 : y0 + bar.shape[0], margin : margin + bar_w, :] = bar
    cv2.rectangle(canvas, (margin, y0), (margin + bar_w - 1, y0 + bar.shape[0] - 1), (210, 210, 210), 1)
    text_y = h + 36
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{min_v:.2f}", (margin, text_y), font, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    max_text = f"{max_v:.2f}"
    (tw, _), _ = cv2.getTextSize(max_text, font, 0.42, 1)
    cv2.putText(canvas, max_text, (max(0, margin + bar_w - tw), text_y), font, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    label_text = label
    (lw, _), _ = cv2.getTextSize(label_text, font, 0.42, 1)
    cv2.putText(canvas, label_text, (max(0, (w - lw) // 2), text_y), font, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas
