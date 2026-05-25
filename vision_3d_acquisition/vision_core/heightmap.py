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
    # Layout (top → bottom inside the appended strip):
    #   - 8 px gap
    #   - 12 px colour gradient bar
    #   - 6 px gap
    #   - min/max value text row (left/right of bar)
    #   - 4 px gap
    #   - centred label text row
    # This 2-row layout avoids the label overlapping the min/max numbers,
    # which happens with the older single-row layout when the label is wide
    # and/or the colorbar is narrow.
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    value_scale = 0.42
    label_scale = 0.46
    bar_thickness = 12
    gap_top = 8
    gap_between = 6
    gap_between_rows = 4
    gap_bottom = 8
    (_, val_h), _ = cv2.getTextSize("0.00", font, value_scale, 1)
    (_, lbl_h), _ = cv2.getTextSize(label or " ", font, label_scale, 1)
    bar_h = (
        gap_top
        + bar_thickness
        + gap_between
        + val_h
        + gap_between_rows
        + lbl_h
        + gap_bottom
    )
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h, :, :] = image
    canvas[h:, :, :] = (18, 18, 18)
    margin = max(24, min(80, w // 12))
    bar_w = max(1, w - (2 * margin))
    gradient = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, bar_w)
    gradient = np.repeat(gradient, bar_thickness, axis=0)
    bar = cv2.applyColorMap(gradient, colormap)
    y0 = h + gap_top
    canvas[y0 : y0 + bar.shape[0], margin : margin + bar_w, :] = bar
    cv2.rectangle(canvas, (margin, y0), (margin + bar_w - 1, y0 + bar.shape[0] - 1), (210, 210, 210), 1)
    text_baseline_values = y0 + bar.shape[0] + gap_between + val_h
    cv2.putText(canvas, f"{min_v:.2f}", (margin, text_baseline_values), font, value_scale, (235, 235, 235), 1, cv2.LINE_AA)
    max_text = f"{max_v:.2f}"
    (tw, _), _ = cv2.getTextSize(max_text, font, value_scale, 1)
    cv2.putText(canvas, max_text, (max(0, margin + bar_w - tw), text_baseline_values), font, value_scale, (235, 235, 235), 1, cv2.LINE_AA)
    if label:
        label_text = label
        (lw, _), _ = cv2.getTextSize(label_text, font, label_scale, 1)
        # If the label is still too wide for the strip, trim with an ellipsis
        # so it never overflows the canvas (still readable in Studio).
        if lw > w - 2 * margin and len(label_text) > 6:
            while lw > w - 2 * margin and len(label_text) > 6:
                label_text = label_text[:-1]
                (lw, _), _ = cv2.getTextSize(label_text + "…", font, label_scale, 1)
            label_text = label_text + "…"
        label_baseline = text_baseline_values + gap_between_rows + lbl_h
        cv2.putText(
            canvas,
            label_text,
            (max(0, (w - lw) // 2), label_baseline),
            font,
            label_scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas
