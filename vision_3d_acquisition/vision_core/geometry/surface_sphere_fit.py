from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame

FEATURE_ALGORITHM_VERSION = "surface_sphere_fit_v1"
MIN_SPHERE_FIT_POINTS = 8


def fit_sphere_least_squares(points_xyz_mm: np.ndarray) -> dict[str, Any]:
    if points_xyz_mm.ndim != 2 or points_xyz_mm.shape[1] != 3:
        return {"valid": False, "reason": "invalid_point_array"}
    if points_xyz_mm.shape[0] < MIN_SPHERE_FIT_POINTS:
        return {
            "valid": False,
            "reason": "insufficient_points",
            "point_count": int(points_xyz_mm.shape[0]),
            "min_points_required": MIN_SPHERE_FIT_POINTS,
        }
    xyz = np.asarray(points_xyz_mm, dtype=np.float64)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    a = np.column_stack((2.0 * x, 2.0 * y, 2.0 * z, np.ones_like(x)))
    b = x * x + y * y + z * z
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except Exception:
        return {"valid": False, "reason": "least_squares_failed"}
    cx, cy, cz, c0 = [float(v) for v in sol.tolist()]
    radius_sq = cx * cx + cy * cy + cz * cz + c0
    if radius_sq <= 0.0:
        return {"valid": False, "reason": "non_positive_radius"}
    radius = float(np.sqrt(radius_sq))
    dist = np.sqrt(np.sum((xyz - np.asarray([cx, cy, cz], dtype=np.float64)) ** 2, axis=1))
    residual = dist - radius
    return {
        "valid": True,
        "center_mm": [round(cx, 4), round(cy, 4), round(cz, 4)],
        "radius_mm": round(radius, 4),
        "rmse_mm": round(float(np.sqrt(np.mean(residual**2))), 4),
        "max_error_mm": round(float(np.max(np.abs(residual))), 4),
        "mean_residual_mm": round(float(np.mean(residual)), 4),
        "point_count": int(xyz.shape[0]),
        "residuals_mm": residual,
    }


def surface_sphere_fit_rmse_mm(points_xyz_mm: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    fit = fit_sphere_least_squares(points_xyz_mm)
    if not fit.get("valid"):
        return None, fit
    return float(fit["rmse_mm"]), fit


def mask_from_contour_px(contour_px: list[list[float]] | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not contour_px or len(contour_px) < 3:
        return mask
    pts = np.asarray(contour_px, dtype=np.int32).reshape(-1, 1, 2)
    filled = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(filled, [pts], 1)
    return filled.astype(bool)


def build_visible_object_points_xyz_mm(
    *,
    normalized_heightmap_mm: np.ndarray,
    frame: HeightmapFrame,
    mask: np.ndarray,
) -> np.ndarray:
    active = np.asarray(mask, dtype=bool) & np.asarray(frame.valid_mask, dtype=bool)
    values = normalized_heightmap_mm[active]
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts_y, pts_x = np.where(active)
    return np.column_stack(
        (
            frame.origin_x_mm + pts_x.astype(np.float64) * float(frame.x_resolution_mm),
            frame.origin_y_mm + pts_y.astype(np.float64) * float(frame.y_resolution_mm),
            np.maximum(values.astype(np.float64), 0.0),
        )
    )
