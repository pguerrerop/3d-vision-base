from __future__ import annotations

from typing import Any

import numpy as np


def measure_cluster(cluster_id: int, cloud: Any) -> dict[str, object]:
    points = np.asarray(cloud.points)
    if points.size == 0:
        center = mins = maxs = dimensions = np.zeros(3, dtype=float)
    else:
        center = points.mean(axis=0)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        dimensions = maxs - mins

    diameter = float(np.linalg.norm(dimensions)) if points.size else None
    return {
        "object_id": cluster_id,
        "class_name": "unknown",
        "confidence": None,
        "point_count": int(points.shape[0]),
        "center_mm": _round_vector(center),
        "dimensions_mm": _round_vector(dimensions),
        "diameter_estimate_mm": round(diameter, 4) if diameter is not None else None,
        "bbox_min_mm": _round_vector(mins),
        "bbox_max_mm": _round_vector(maxs),
        "diameter_mm": None,
        "sphericity_score": None,
        "fit_rmse_mm": None,
    }


def measure_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, object]]:
    return [measure_cluster(int(cluster["cluster_id"]), cluster["cloud"]) for cluster in clusters]


def _round_vector(vector: np.ndarray) -> tuple[float, float, float]:
    values = [round(float(value), 4) for value in vector.tolist()]
    return (values[0], values[1], values[2])
