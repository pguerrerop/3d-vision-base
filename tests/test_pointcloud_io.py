from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_3d_acquisition.processing.pointcloud_io import load_point_cloud_fast, save_point_cloud_npz


def test_save_load_npz_preserves_point_count_and_bounds(tmp_path: Path) -> None:
    points = np.asarray(
        [
            [-1.0, 2.0, 0.5],
            [3.0, -4.0, 5.5],
            [0.25, 0.5, 1.5],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "point_cloud.npz"

    save_point_cloud_npz(points, path)
    loaded = load_point_cloud_fast(path)
    loaded_points = np.asarray(loaded.points)

    assert len(loaded.points) == len(points)
    np.testing.assert_allclose(loaded_points.min(axis=0), points.min(axis=0))
    np.testing.assert_allclose(loaded_points.max(axis=0), points.max(axis=0))
