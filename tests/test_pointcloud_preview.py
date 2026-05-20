from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.processing.pointcloud_preview import compute_point_cloud_stats


def test_compute_point_cloud_stats_for_ascii_ply(tmp_path: Path) -> None:
    ply = tmp_path / "points.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "1.0 2.0 3.0",
                "4.0 6.0 9.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = compute_point_cloud_stats(ply)

    assert stats["point_count"] == 2
    assert stats["has_colors"] is False
    assert stats["has_normals"] is False
    assert stats["min_bound"] == [1.0, 2.0, 3.0]
    assert stats["max_bound"] == [4.0, 6.0, 9.0]
    assert stats["extent"] == [3.0, 4.0, 6.0]
    assert stats["file_size_bytes"] > 0
