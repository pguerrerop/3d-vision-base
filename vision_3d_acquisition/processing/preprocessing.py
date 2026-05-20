from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    voxel_size: float = 1.0
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0


def voxel_downsample(pcd: Any, voxel_size: float = 1.0) -> Any:
    if voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_outliers(
    pcd: Any,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Any:
    if len(pcd.points) < max(nb_neighbors, 3):
        return pcd
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered


def preprocess_point_cloud(pcd: Any, config: PreprocessingConfig) -> Any:
    downsampled = voxel_downsample(pcd, config.voxel_size)
    return remove_outliers(
        downsampled,
        nb_neighbors=config.outlier_nb_neighbors,
        std_ratio=config.outlier_std_ratio,
    )
