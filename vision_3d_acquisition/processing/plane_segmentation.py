from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlaneSegmentationConfig:
    distance_threshold: float = 1.5
    ransac_n: int = 3
    num_iterations: int = 1000


def segment_dominant_plane(
    pcd: Any,
    distance_threshold: float = 1.5,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> dict[str, Any]:
    point_count = len(pcd.points)
    if point_count < ransac_n:
        return {
            "plane_model": None,
            "plane_cloud": pcd.select_by_index([]),
            "foreground_cloud": pcd,
            "inlier_indices": [],
            "outlier_indices": list(range(point_count)),
        }

    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    inlier_set = set(inlier_indices)
    outlier_indices = [index for index in range(point_count) if index not in inlier_set]
    return {
        "plane_model": [float(value) for value in plane_model],
        "plane_cloud": pcd.select_by_index(inlier_indices),
        "foreground_cloud": pcd.select_by_index(inlier_indices, invert=True),
        "inlier_indices": list(inlier_indices),
        "outlier_indices": outlier_indices,
    }
