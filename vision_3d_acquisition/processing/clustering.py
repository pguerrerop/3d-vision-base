from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClusteringConfig:
    dbscan_eps: float = 3.0
    dbscan_min_points: int = 30
    min_cluster_points: int = 100


def cluster_foreground(
    pcd: Any,
    dbscan_eps: float = 3.0,
    dbscan_min_points: int = 30,
    min_cluster_points: int = 100,
) -> list[dict[str, Any]]:
    if len(pcd.points) == 0:
        return []

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=dbscan_eps,
            min_points=dbscan_min_points,
            print_progress=False,
        )
    )
    clusters: list[dict[str, Any]] = []
    cluster_id = 1
    for label in sorted(set(int(value) for value in labels if int(value) >= 0)):
        indices = np.where(labels == label)[0].tolist()
        if len(indices) < min_cluster_points:
            continue
        clusters.append(
            {
                "cluster_id": cluster_id,
                "point_indices": indices,
                "cloud": pcd.select_by_index(indices),
            }
        )
        cluster_id += 1
    return clusters
