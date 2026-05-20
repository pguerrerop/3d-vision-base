from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PointCloudStatsDict = dict[str, int | bool | list[float]]


def load_point_cloud(path: Path) -> Any:
    """Load a point cloud with Open3D.

    The import stays lazy so API/UI code can still import this module in
    environments where Open3D is not initialized until processing time.
    """
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        return point_cloud
    return point_cloud


def compute_point_cloud_stats(path: Path) -> dict[str, int | bool | list[float]]:
    try:
        point_cloud = load_point_cloud(path)
        points = np.asarray(point_cloud.points)
        has_colors = bool(point_cloud.has_colors())
        has_normals = bool(point_cloud.has_normals())
    except Exception:
        points, has_colors, has_normals = _read_ascii_ply_points(path)

    if points.size == 0:
        min_bound = max_bound = extent = np.zeros(3, dtype=float)
    else:
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        extent = max_bound - min_bound

    return {
        "point_count": int(points.shape[0]),
        "has_colors": has_colors,
        "has_normals": has_normals,
        "min_bound": _round_vector(min_bound),
        "max_bound": _round_vector(max_bound),
        "extent": _round_vector(extent),
        "file_size_bytes": path.stat().st_size,
    }


def render_point_cloud_preview(path: Path, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    points = np.empty((0, 3), dtype=float)
    try:
        point_cloud = load_point_cloud(path)
        points = np.asarray(point_cloud.points)
        if not point_cloud.is_empty() and _use_open3d_offscreen():
            _render_open3d_preview(point_cloud, output_png)
            return
    except Exception:
        points, _, _ = _read_ascii_ply_points(path)

    if points.size == 0:
        points, _, _ = _read_ascii_ply_points(path)
    _render_matplotlib_preview(points, output_png)


def _use_open3d_offscreen() -> bool:
    return os.environ.get("VISION_USE_OPEN3D_OFFSCREEN", "").strip().lower() in {"1", "true", "yes", "on"}


def _render_open3d_preview(point_cloud: Any, output_png: Path) -> None:
    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
    try:
        scene = renderer.scene
        scene.set_background([0.02, 0.025, 0.035, 1.0])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0
        scene.add_geometry("input_point_cloud", point_cloud, material)
        bounds = point_cloud.get_axis_aligned_bounding_box()
        scene.camera.look_at(bounds.get_center(), bounds.get_center() + [0.0, -1.5, 0.8], [0.0, 0.0, 1.0])
        image = renderer.render_to_image()
        o3d.io.write_image(str(output_png), image)
    finally:
        renderer.release_resources()


def _render_matplotlib_preview(points: np.ndarray, output_png: Path) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "vision_3d_acquisition_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#111827")
    axis = fig.add_subplot(111, projection="3d", facecolor="#111827")
    axis.set_xlabel("X mm", color="#d8dee9")
    axis.set_ylabel("Y mm", color="#d8dee9")
    axis.set_zlabel("Z mm", color="#d8dee9")
    axis.tick_params(colors="#d8dee9")
    axis.grid(True, color="#334155", linewidth=0.5)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor("#111827")
        pane.set_edgecolor("#475569")

    if points.size:
        sample = points
        if len(sample) > 50000:
            indices = np.linspace(0, len(sample) - 1, 50000, dtype=int)
            sample = sample[indices]
        color_values = sample[:, 2] if sample.shape[1] >= 3 else "#68d391"
        axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=color_values, cmap="viridis", s=2, alpha=0.85)
        _set_equal_axes(axis, sample)
    else:
        axis.text2D(0.5, 0.5, "No points in input point cloud", transform=axis.transAxes, color="#e5edf5", ha="center")

    fig.tight_layout()
    fig.savefig(output_png, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def _read_ascii_ply_points(path: Path) -> tuple[np.ndarray, bool, bool]:
    has_colors = False
    has_normals = False
    vertex_count = 0
    properties: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_end = 0
    in_vertex = False
    for index, line in enumerate(lines):
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) >= 3:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts and parts[0] == "element" and parts[1:2] != ["vertex"]:
            in_vertex = False
        if in_vertex and parts[:1] == ["property"] and len(parts) >= 3:
            properties.append(parts[-1])
        if line == "end_header":
            header_end = index + 1
            break

    has_colors = {"red", "green", "blue"}.issubset(set(properties))
    has_normals = {"nx", "ny", "nz"}.issubset(set(properties))
    rows: list[list[float]] = []
    for line in lines[header_end : header_end + vertex_count]:
        values = line.split()
        if len(values) < 3:
            continue
        try:
            rows.append([float(values[0]), float(values[1]), float(values[2])])
        except ValueError:
            continue
    points = np.asarray(rows, dtype=float).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=float)
    return points, has_colors, has_normals


def _round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in vector.tolist()]


def _set_equal_axes(axis: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 1.0)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)
