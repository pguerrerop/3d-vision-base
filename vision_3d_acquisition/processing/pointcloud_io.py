from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_point_cloud(path: Path) -> Any:
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        raise ValueError(f"Point cloud has no points: {path}")
    return point_cloud


def load_point_cloud_fast(path: Path) -> Any:
    if path.suffix.lower() == ".npz":
        return _load_point_cloud_npz(path)
    return load_point_cloud(path)


def save_point_cloud_npz(pcd_or_points: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = _points_array(pcd_or_points).astype(np.float32, copy=False)
    payload: dict[str, np.ndarray] = {"points": points}
    if hasattr(pcd_or_points, "has_colors") and pcd_or_points.has_colors():
        payload["colors"] = np.asarray(pcd_or_points.colors, dtype=np.float32)
    if hasattr(pcd_or_points, "has_normals") and pcd_or_points.has_normals():
        payload["normals"] = np.asarray(pcd_or_points.normals, dtype=np.float32)
    np.savez(path, **payload)


def save_point_cloud(path: Path, pcd: Any) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise OSError(f"Failed to write point cloud: {path}")


def _load_point_cloud_npz(path: Path) -> Any:
    import open3d as o3d

    with np.load(path) as payload:
        if "points" not in payload:
            raise ValueError(f"NPZ point cloud missing 'points' array: {path}")
        points = np.asarray(payload["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points array with shape Nx3 in {path}, got {points.shape}")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if "colors" in payload:
            colors = np.asarray(payload["colors"], dtype=np.float64)
            if colors.shape == points.shape:
                point_cloud.colors = o3d.utility.Vector3dVector(colors)
        if "normals" in payload:
            normals = np.asarray(payload["normals"], dtype=np.float64)
            if normals.shape == points.shape:
                point_cloud.normals = o3d.utility.Vector3dVector(normals)
    if point_cloud.is_empty():
        raise ValueError(f"Point cloud has no points: {path}")
    return point_cloud


def _points_array(pcd_or_points: Any) -> np.ndarray:
    if hasattr(pcd_or_points, "points"):
        return np.asarray(pcd_or_points.points)
    points = np.asarray(pcd_or_points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point array with shape Nx3, got {points.shape}")
    return points
