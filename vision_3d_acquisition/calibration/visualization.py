from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PLANE_COLORS = ["#38bdf8", "#f97316", "#22c55e", "#f43f5e", "#eab308", "#a78bfa", "#14b8a6"]


def generate_plane_visualizations(
    plane_candidates: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = output_dir / "plane_candidates.png"
    _render_matplotlib(
        [(candidate["_cloud"], PLANE_COLORS[index % len(PLANE_COLORS)], candidate["plane_id"]) for index, candidate in enumerate(plane_candidates)],
        combined,
        "Plane candidates",
    )
    files = {"preview_image": combined.name}
    for index, candidate in enumerate(plane_candidates):
        path = output_dir / f"{candidate['plane_id']}.png"
        _render_matplotlib(
            [(candidate["_cloud"], PLANE_COLORS[index % len(PLANE_COLORS)], candidate["plane_id"])],
            path,
            candidate["plane_id"],
        )
        files[candidate["plane_id"]] = path.name
    return files


def _render_matplotlib(clouds: list[tuple[Any, str, str]], output_png: Path, title: str) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "vision_3d_acquisition_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#101820")
    axis = fig.add_subplot(111, projection="3d", facecolor="#101820")
    axis.set_title(title, color="#eef4fb", pad=16)
    axis.set_xlabel("X mm", color="#d8dee9")
    axis.set_ylabel("Y mm", color="#d8dee9")
    axis.set_zlabel("Z mm", color="#d8dee9")
    axis.tick_params(colors="#d8dee9")
    axis.grid(True, color="#334155", linewidth=0.5)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor("#101820")
        pane.set_edgecolor("#475569")

    all_points: list[np.ndarray] = []
    for cloud, color, label in clouds:
        points = np.asarray(cloud.points)
        if len(points) == 0:
            continue
        sample = _sample_points(points)
        all_points.append(sample)
        axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=color, s=4, alpha=0.9, label=label)
        center = sample.mean(axis=0)
        axis.text(center[0], center[1], center[2], label, color="#f8fafc", fontsize=10)

    if all_points:
        _set_equal_axes(axis, np.vstack(all_points))
        legend = axis.legend(facecolor="#17202a", edgecolor="#475569")
        for text in legend.get_texts():
            text.set_color("#eef4fb")
    else:
        axis.text2D(0.5, 0.5, "No planes detected", transform=axis.transAxes, color="#eef4fb", ha="center")

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

