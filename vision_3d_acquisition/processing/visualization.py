from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def render_point_cloud_preview(pcd: Any, output_png: Path) -> None:
    _render_clouds(
        [(pcd, "#58a6ff")],
        output_png,
        title="Input point cloud",
    )


def render_plane_segmentation(plane_cloud: Any, foreground_cloud: Any, output_png: Path) -> None:
    _render_clouds(
        [(plane_cloud, "#9ca3af"), (foreground_cloud, "#34d399")],
        output_png,
        title="Plane segmentation: background gray, foreground green",
    )


def render_foreground(foreground_cloud: Any, output_png: Path) -> None:
    _render_clouds(
        [(foreground_cloud, "#34d399")],
        output_png,
        title="Foreground after dominant plane removal",
    )


def render_clusters(clusters: list[dict[str, Any]], output_png: Path) -> None:
    palette = [
        "#38bdf8",
        "#f97316",
        "#a78bfa",
        "#22c55e",
        "#f43f5e",
        "#eab308",
        "#14b8a6",
        "#fb7185",
    ]
    clouds = [(cluster["cloud"], palette[index % len(palette)]) for index, cluster in enumerate(clusters)]
    _render_clouds(clouds, output_png, title="Foreground clusters")


def render_calibrated_planes(belt_cloud: Any, ignored_cloud: Any, candidate_cloud: Any, output_png: Path) -> None:
    _render_clouds(
        [(belt_cloud, "#38bdf8"), (ignored_cloud, "#94a3b8"), (candidate_cloud, "#22c55e")],
        output_png,
        title="Calibrated planes and candidate foreground (mm)",
    )


def render_rejected_points(rejected_cloud: Any, output_png: Path) -> None:
    _render_clouds(
        [(rejected_cloud, "#f97316")],
        output_png,
        title="Rejected calibrated points (mm)",
    )


def render_filtered_clusters(clusters: list[dict[str, Any]], output_png: Path) -> None:
    palette = [
        "#38bdf8",
        "#22c55e",
        "#eab308",
        "#a78bfa",
        "#f43f5e",
        "#14b8a6",
        "#fb7185",
    ]
    clouds = [(cluster["cloud"], palette[index % len(palette)]) for index, cluster in enumerate(clusters)]
    _render_clouds(clouds, output_png, title="Filtered kept clusters (mm)")


def render_belt_polygon_topview(
    candidate_cloud: Any,
    rejected_cloud: Any,
    polygon_xy: list[tuple[float, float]],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.gettempdir()) / "vision_3d_acquisition_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 7), facecolor="#111827")
    axis.set_facecolor("#111827")
    axis.set_title("Belt polygon top view (XY mm)", color="#e5edf5", pad=16)
    axis.set_xlabel("X mm", color="#d8dee9")
    axis.set_ylabel("Y mm", color="#d8dee9")
    axis.tick_params(colors="#d8dee9")
    axis.grid(True, color="#334155", linewidth=0.5)

    all_xy: list[np.ndarray] = []
    for cloud, color, label in ((candidate_cloud, "#22c55e", "candidate"), (rejected_cloud, "#f97316", "rejected")):
        points = np.asarray(cloud.points)
        if len(points) == 0:
            continue
        sample = _sample_points(points)[:, :2]
        all_xy.append(sample)
        axis.scatter(sample[:, 0], sample[:, 1], c=color, s=4, alpha=0.75, label=label)

    if len(polygon_xy) >= 3:
        polygon = np.asarray(polygon_xy, dtype=float)
        closed = np.vstack([polygon, polygon[0]])
        all_xy.append(polygon)
        axis.plot(closed[:, 0], closed[:, 1], color="#38bdf8", linewidth=2.8, label="belt ROI")
        axis.fill(polygon[:, 0], polygon[:, 1], color="#38bdf8", alpha=0.16)

    if all_xy:
        stacked = np.vstack(all_xy)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        centers = (mins + maxs) / 2
        radius = max(float((maxs - mins).max()) / 2, 1.0)
        axis.set_xlim(centers[0] - radius, centers[0] + radius)
        axis.set_ylim(centers[1] - radius, centers[1] + radius)
        axis.set_aspect("equal", adjustable="box")
        legend = axis.legend(facecolor="#17202a", edgecolor="#475569")
        for text in legend.get_texts():
            text.set_color("#eef4fb")
    else:
        axis.text(0.5, 0.5, "No calibrated points to display", transform=axis.transAxes, color="#e5edf5", ha="center")

    fig.tight_layout()
    fig.savefig(output_png, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_clouds(clouds: list[tuple[Any, str]], output_png: Path, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if _use_open3d_offscreen():
        try:
            _render_open3d_clouds(clouds, output_png)
            return
        except Exception:
            pass
    _render_matplotlib_clouds(clouds, output_png, title)


def _use_open3d_offscreen() -> bool:
    return os.environ.get("VISION_USE_OPEN3D_OFFSCREEN", "").strip().lower() in {"1", "true", "yes", "on"}


def _render_open3d_clouds(clouds: list[tuple[Any, str]], output_png: Path) -> None:
    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
    try:
        scene = renderer.scene
        scene.set_background([0.02, 0.025, 0.035, 1.0])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0
        all_points: list[np.ndarray] = []
        for index, (cloud, color) in enumerate(clouds):
            painted = _copy_with_color(cloud, _hex_to_rgb(color))
            points = np.asarray(painted.points)
            if len(points) == 0:
                continue
            all_points.append(points)
            scene.add_geometry(f"cloud_{index}", painted, material)
        if not all_points:
            raise ValueError("No points to render")
        stacked = np.vstack(all_points)
        center = stacked.mean(axis=0)
        extent = np.ptp(stacked, axis=0)
        distance = max(float(np.linalg.norm(extent)) * 1.4, 10.0)
        scene.camera.look_at(center, center + [distance, -distance, distance * 0.65], [0.0, 0.0, 1.0])
        image = renderer.render_to_image()
        o3d.io.write_image(str(output_png), image)
    finally:
        renderer.release_resources()


def _render_matplotlib_clouds(clouds: list[tuple[Any, str]], output_png: Path, title: str) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "vision_3d_acquisition_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#111827")
    axis = fig.add_subplot(111, projection="3d", facecolor="#111827")
    axis.set_title(title, color="#e5edf5", pad=16)
    axis.set_xlabel("X mm", color="#d8dee9")
    axis.set_ylabel("Y mm", color="#d8dee9")
    axis.set_zlabel("Z mm", color="#d8dee9")
    axis.tick_params(colors="#d8dee9")
    axis.grid(True, color="#334155", linewidth=0.5)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor("#111827")
        pane.set_edgecolor("#475569")

    all_points: list[np.ndarray] = []
    for cloud, color in clouds:
        points = np.asarray(cloud.points)
        if len(points) == 0:
            continue
        sample = _sample_points(points)
        all_points.append(sample)
        axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=color, s=3, alpha=0.88)

    if all_points:
        _set_equal_axes(axis, np.vstack(all_points))
    else:
        axis.text2D(0.5, 0.5, "No points to display", transform=axis.transAxes, color="#e5edf5", ha="center")

    fig.tight_layout()
    fig.savefig(output_png, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def _sample_points(points: np.ndarray, max_points: int = 50000) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def _set_equal_axes(axis: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 1.0)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


def _copy_with_color(cloud: Any, color: tuple[float, float, float]) -> Any:
    import open3d as o3d

    copied = o3d.geometry.PointCloud(cloud)
    copied.paint_uniform_color(color)
    return copied


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    stripped = value.lstrip("#")
    return (
        int(stripped[0:2], 16) / 255.0,
        int(stripped[2:4], 16) / 255.0,
        int(stripped[4:6], 16) / 255.0,
    )
