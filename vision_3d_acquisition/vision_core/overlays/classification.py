from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ClassificationOverlayObject:
    object_id: int
    object_key: str
    class_label: str
    confidence: float
    status: str
    contour: list[list[int]]
    bounding_box: dict[str, float]
    ellipse: dict[str, float] | None
    measurements: dict[str, Any]
    annotations: list[str]


_STATUS_COLORS = {
    "accepted": (30, 198, 118),
    "rejected": (77, 89, 255),
    "suspicious": (51, 186, 255),
}


def _status_from_label(label: str) -> str:
    if label == "Bola buena":
        return "accepted"
    if label == "Scrap de Bola":
        return "rejected"
    return "suspicious"


def build_classification_overlay_objects(
    classes: list[dict[str, Any]],
    *,
    candidates_by_key: dict[str, dict[str, Any]],
) -> list[ClassificationOverlayObject]:
    objects: list[ClassificationOverlayObject] = []
    for row in classes:
        object_key = str(row.get("object_key", "")).strip()
        source = candidates_by_key.get(object_key) or {}
        contour = source.get("contour")
        contour_points = _normalize_points(contour)
        bbox = _normalize_bbox(source.get("bbox"))
        ellipse = source.get("ellipse") if isinstance(source.get("ellipse"), dict) else None
        measurements = {
            "diameter_px": _num((row.get("geometry") or {}).get("diameter_px")),
            "axis_ratio": _num((row.get("geometry") or {}).get("axis_ratio")),
            "circularity": _num((row.get("geometry") or {}).get("circularity")),
            "area_px": _num((row.get("geometry") or {}).get("area_px")),
            "fit_rmse": _num(row.get("fit_rmse")),
        }
        objects.append(
            ClassificationOverlayObject(
                object_id=int(row.get("object_id", 0) or 0),
                object_key=object_key or f"object_{int(row.get('object_id', 0) or 0):03d}",
                class_label=str(row.get("class_label", "Chatarra")),
                confidence=float(_num(row.get("confidence")) or 0.0),
                status=_status_from_label(str(row.get("class_label", "Chatarra"))),
                contour=contour_points,
                bounding_box=bbox,
                ellipse=_normalize_ellipse(ellipse),
                measurements=measurements,
                annotations=[str(x) for x in list(row.get("decision_reasons") or [])],
            )
        )
    return objects


def render_classification_overlay(
    source_rgb: np.ndarray,
    *,
    objects: list[ClassificationOverlayObject],
    draw_geometry: bool = True,
) -> np.ndarray:
    canvas = source_rgb.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for item in sorted(objects, key=lambda obj: obj.object_id):
        color = _STATUS_COLORS.get(item.status, _STATUS_COLORS["suspicious"])
        contour = np.array(item.contour, dtype=np.int32).reshape((-1, 1, 2)) if item.contour else None
        if contour is not None and contour.size:
            cv2.polylines(canvas, [contour], True, color, 2, lineType=cv2.LINE_AA)
        bbox = item.bounding_box
        x = int(round(float(bbox.get("x", 0.0))))
        y = int(round(float(bbox.get("y", 0.0))))
        w = int(round(float(bbox.get("width", 0.0))))
        h = int(round(float(bbox.get("height", 0.0))))
        if w > 0 and h > 0:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
        if draw_geometry and item.ellipse:
            ellipse = item.ellipse
            cx = int(round(float(ellipse.get("cx", 0.0))))
            cy = int(round(float(ellipse.get("cy", 0.0))))
            rx = max(1, int(round(float(ellipse.get("rx", 1.0)))))
            ry = max(1, int(round(float(ellipse.get("ry", 1.0)))))
            angle = float(ellipse.get("rotation_deg", 0.0))
            cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0, 360, color, 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (cx - rx, cy), (cx + rx, cy), color, 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (cx, cy - ry), (cx, cy + ry), color, 1, lineType=cv2.LINE_AA)
    return canvas


def _normalize_points(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    points: list[list[int]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = _num(point[0])
        y = _num(point[1])
        if x is None or y is None:
            continue
        points.append([int(round(x)), int(round(y))])
    return points


def _normalize_bbox(value: Any) -> dict[str, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return {
            "x": float(_num(value[0]) or 0.0),
            "y": float(_num(value[1]) or 0.0),
            "width": float(_num(value[2]) or 0.0),
            "height": float(_num(value[3]) or 0.0),
        }
    if isinstance(value, dict):
        return {
            "x": float(_num(value.get("x")) or 0.0),
            "y": float(_num(value.get("y")) or 0.0),
            "width": float(_num(value.get("width")) or 0.0),
            "height": float(_num(value.get("height")) or 0.0),
        }
    return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}


def _normalize_ellipse(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    cx = _num(value.get("cx"))
    cy = _num(value.get("cy"))
    rx = _num(value.get("rx"))
    ry = _num(value.get("ry"))
    if cx is None or cy is None or rx is None or ry is None:
        return None
    return {
        "cx": float(cx),
        "cy": float(cy),
        "rx": float(rx),
        "ry": float(ry),
        "rotation_deg": float(_num(value.get("rotation_deg")) or 0.0),
    }


def _num(value: Any) -> float | None:
    parsed = float(value) if isinstance(value, (int, float)) else None
    if parsed is not None and np.isfinite(parsed):
        return parsed
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if np.isfinite(parsed) else None
    return None
