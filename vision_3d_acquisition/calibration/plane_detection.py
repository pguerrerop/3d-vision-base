from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def detect_plane_candidates(
    pcd: Any,
    max_planes: int = 5,
    min_plane_points: int = 5000,
    distance_threshold_mm: float = 1.5,
) -> list[dict[str, Any]]:
    """Detect dominant planes with iterative RANSAC and inlier removal."""
    remaining = pcd
    candidates: list[dict[str, Any]] = []

    for index in range(max_planes):
        if len(remaining.points) < min_plane_points:
            break
        plane_model, inlier_indices = remaining.segment_plane(
            distance_threshold=distance_threshold_mm,
            ransac_n=3,
            num_iterations=1000,
        )
        if len(inlier_indices) < min_plane_points:
            break

        plane_cloud = remaining.select_by_index(inlier_indices)
        points = np.asarray(plane_cloud.points)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extent = maxs - mins
        centroid = points.mean(axis=0)
        normal = np.asarray(plane_model[:3], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 0:
            normal = normal / norm

        candidates.append(
            {
                "plane_id": f"plane_{index + 1}",
                "plane_model": [float(value) for value in plane_model],
                "point_count": int(points.shape[0]),
                "centroid_mm": _round_vector(centroid),
                "bbox_min_mm": _round_vector(mins),
                "bbox_max_mm": _round_vector(maxs),
                "extent_mm": _round_vector(extent),
                "avg_z_mm": round(float(points[:, 2].mean()), 4),
                "normal": _round_vector(normal),
                "roi_polygon_xy_mm": _convex_hull_xy(points[:, :2], mins[:2], maxs[:2]),
                "_cloud": plane_cloud,
            }
        )
        remaining = remaining.select_by_index(inlier_indices, invert=True)

    return candidates


def public_plane_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def _round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in vector.tolist()]


def _convex_hull_xy(points_xy: np.ndarray, bbox_min_xy: np.ndarray, bbox_max_xy: np.ndarray) -> list[list[float]]:
    unique = np.unique(points_xy, axis=0)
    if len(unique) >= 3:
        try:
            hull = ConvexHull(unique)
            hull_points = unique[hull.vertices]
            return [_round_xy(point) for point in hull_points]
        except QhullError:
            pass
    min_x, min_y = bbox_min_xy.tolist()
    max_x, max_y = bbox_max_xy.tolist()
    if min_x == max_x and min_y == max_y:
        return [[round(float(min_x), 4), round(float(min_y), 4)]]
    return [
        [round(float(min_x), 4), round(float(min_y), 4)],
        [round(float(max_x), 4), round(float(min_y), 4)],
        [round(float(max_x), 4), round(float(max_y), 4)],
        [round(float(min_x), 4), round(float(max_y), 4)],
    ]


def _round_xy(point: np.ndarray) -> list[float]:
    return [round(float(point[0]), 4), round(float(point[1]), 4)]
