from __future__ import annotations

from typing import Any

import numpy as np

from vision_3d_acquisition.calibration.calibration_models import PlaneCalibration, SystemCalibration


def distance_to_plane(points: np.ndarray, plane_model: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = [float(value) for value in plane_model]
    normal_norm = max(float(np.linalg.norm([a, b, c])), 1e-9)
    return (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_norm


def points_inside_polygon_xy(points: np.ndarray, polygon_xy: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=float)
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    return _points_in_polygon(points[:, :2], polygon)


def filter_by_labeled_planes(
    pcd: Any,
    calibration: SystemCalibration,
    distance_threshold_mm: float = 1.5,
) -> dict[str, Any]:
    belt = _require_belt(calibration)
    points = np.asarray(pcd.points)
    point_count = len(points)
    if point_count == 0:
        empty = pcd.select_by_index([])
        return {
            "belt_plane": belt,
            "ignored_planes": calibration.ignored_planes(),
            "plane_model": tuple(float(value) for value in belt.plane_model),
            "belt_reference_points": empty,
            "ignored_plane_points": empty,
            "candidate_foreground_points": empty,
            "rejected_points": empty,
            "foreground_cloud": empty,
            "masks": {
                "belt_reference": np.zeros(0, dtype=bool),
                "ignored_plane": np.zeros(0, dtype=bool),
                "candidate_foreground": np.zeros(0, dtype=bool),
                "rejected": np.zeros(0, dtype=bool),
                "inside_belt_roi": np.zeros(0, dtype=bool),
            },
            "heights_above_belt_mm": np.zeros(0, dtype=float),
            "statistics": _stats(calibration, 0, 0, 0, 0),
        }

    belt_model = _oriented_belt_plane_model(belt.plane_model)
    belt_distances = distance_to_plane(points, belt_model)
    belt_reference_mask = np.abs(belt_distances) <= distance_threshold_mm

    ignored_mask = np.zeros(point_count, dtype=bool)
    for ignored in calibration.ignored_planes():
        ignored_mask |= np.abs(distance_to_plane(points, ignored.plane_model)) <= distance_threshold_mm

    heights = height_above_belt(points, belt_model)
    inside_roi = _inside_belt_roi(points, belt)
    object_filter = calibration.object_filter
    candidate_mask = (
        ~ignored_mask
        & inside_roi
        & (heights >= object_filter.min_height_above_belt_mm)
        & (heights <= object_filter.max_height_above_belt_mm)
    )
    rejected_mask = ~candidate_mask & ~ignored_mask & ~belt_reference_mask

    return {
        "belt_plane": belt,
        "ignored_planes": calibration.ignored_planes(),
        "plane_model": tuple(float(value) for value in belt_model),
        "belt_reference_points": pcd.select_by_index(np.where(belt_reference_mask)[0].tolist()),
        "ignored_plane_points": pcd.select_by_index(np.where(ignored_mask)[0].tolist()),
        "candidate_foreground_points": pcd.select_by_index(np.where(candidate_mask)[0].tolist()),
        "rejected_points": pcd.select_by_index(np.where(rejected_mask)[0].tolist()),
        "foreground_cloud": pcd.select_by_index(np.where(candidate_mask)[0].tolist()),
        "masks": {
            "belt_reference": belt_reference_mask,
            "ignored_plane": ignored_mask,
            "candidate_foreground": candidate_mask,
            "rejected": rejected_mask,
            "inside_belt_roi": inside_roi,
        },
        "heights_above_belt_mm": heights,
        "statistics": _stats(
            calibration,
            point_count,
            int(np.count_nonzero(ignored_mask)),
            int(np.count_nonzero(candidate_mask)),
            int(np.count_nonzero(rejected_mask)),
        ),
    }


def evaluate_clusters_with_calibration(
    clusters: list[dict[str, Any]],
    calibration: SystemCalibration,
    min_cluster_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    belt = _require_belt(calibration)
    belt_model = _oriented_belt_plane_model(belt.plane_model)
    object_filter = calibration.object_filter
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for cluster in clusters:
        points = np.asarray(cluster["cloud"].points)
        reason: str | None = None
        if len(points) < min_cluster_points:
            reason = "point_count_below_minimum"
        heights = height_above_belt(points, belt_model) if len(points) else np.zeros(0, dtype=float)
        inside = _inside_belt_roi(points, belt) if len(points) else np.zeros(0, dtype=bool)
        center = points.mean(axis=0) if len(points) else np.zeros(3, dtype=float)
        center_inside = bool(_inside_belt_roi(center.reshape(1, 3), belt)[0]) if len(points) else False
        fraction_inside = float(np.count_nonzero(inside) / len(points)) if len(points) else 0.0

        if reason is None and object_filter.require_center_inside_belt and not center_inside:
            reason = "center_outside_belt_polygon"
        if reason is None and fraction_inside < object_filter.min_fraction_points_inside_belt:
            reason = "insufficient_points_inside_belt_polygon"
        if reason is None and len(heights) and float(heights.min()) < object_filter.min_height_above_belt_mm:
            reason = "cluster_below_min_height"
        if reason is None and len(heights) and float(heights.max()) > object_filter.max_height_above_belt_mm:
            reason = "cluster_above_max_height"

        cluster["calibration_metrics"] = {
            "height_above_belt_mm": _height_stats(heights),
            "fraction_points_inside_belt": round(fraction_inside, 4),
            "filter_status": "rejected" if reason else "kept",
            "filter_reason": reason,
        }
        if reason:
            rejected.append(cluster)
        else:
            kept.append(cluster)

    for index, cluster in enumerate(kept, start=1):
        cluster["cluster_id"] = index
    for index, cluster in enumerate(rejected, start=1):
        cluster["cluster_id"] = index
    return kept, rejected


def height_above_belt(points: np.ndarray, plane_model: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = [float(value) for value in plane_model]
    if abs(c) >= 1e-9:
        plane_z = -((a * points[:, 0] + b * points[:, 1] + d) / c)
        return points[:, 2] - plane_z
    return distance_to_plane(points, _oriented_belt_plane_model(plane_model))


def _require_belt(calibration: SystemCalibration) -> PlaneCalibration:
    belts = [plane for plane in calibration.planes if plane.label == "belt"]
    if len(belts) != 1:
        raise ValueError(f"Expected exactly one belt plane in calibration {calibration.calibration_id}, found {len(belts)}")
    return belts[0]


def _oriented_belt_plane_model(plane_model: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    a, b, c, d = [float(value) for value in plane_model]
    if c < 0:
        return (-a, -b, -c, -d)
    return (a, b, c, d)


def _inside_belt_roi(points: np.ndarray, belt: PlaneCalibration) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if len(belt.roi_polygon_xy_mm) >= 3:
        return points_inside_polygon_xy(points, belt.roi_polygon_xy_mm)
    bbox_min = np.asarray(belt.bbox_min_mm, dtype=float)
    bbox_max = np.asarray(belt.bbox_max_mm, dtype=float)
    return (
        (points[:, 0] >= bbox_min[0])
        & (points[:, 0] <= bbox_max[0])
        & (points[:, 1] >= bbox_min[1])
        & (points[:, 1] <= bbox_max[1])
    )


def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = np.zeros(len(points_xy), dtype=bool)
    on_boundary = np.zeros(len(points_xy), dtype=bool)
    j = len(polygon_xy) - 1
    for i in range(len(polygon_xy)):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        denominator = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        intersects = ((yi > y) != (yj > y)) & (x < ((xj - xi) * (y - yi) / denominator + xi))
        inside ^= intersects
        on_boundary |= _points_on_segment(points_xy, np.asarray([xj, yj]), np.asarray([xi, yi]))
        j = i
    return inside | on_boundary


def _points_on_segment(points_xy: np.ndarray, start: np.ndarray, end: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= tolerance:
        return np.linalg.norm(points_xy - start, axis=1) <= tolerance
    relative = points_xy - start
    cross = np.abs(relative[:, 0] * segment[1] - relative[:, 1] * segment[0])
    dot = relative @ segment
    return (cross <= tolerance) & (dot >= -tolerance) & (dot <= length_sq + tolerance)


def _height_stats(heights: np.ndarray) -> dict[str, float]:
    if len(heights) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(float(heights.min()), 4),
        "max": round(float(heights.max()), 4),
        "mean": round(float(heights.mean()), 4),
    }


def _stats(
    calibration: SystemCalibration,
    input_points: int,
    ignored_plane_points: int,
    candidate_foreground_points: int,
    rejected_points: int,
) -> dict[str, Any]:
    return {
        "belt_plane_id": _require_belt(calibration).plane_id,
        "ignored_plane_ids": [plane.plane_id for plane in calibration.ignored_planes()],
        "input_points": input_points,
        "ignored_plane_points": ignored_plane_points,
        "candidate_foreground_points": candidate_foreground_points,
        "rejected_points": rejected_points,
        "kept_objects": 0,
        "rejected_objects": 0,
    }
