from __future__ import annotations

import numpy as np

from vision_3d_acquisition.processes.executor import PipelineExecutor2D
from vision_3d_acquisition.processes.models import PipelineStepConfig


def test_ellipse_metrics_include_mm_fields_with_active_2d_calibration() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    contour = cv_points_circle(150, 150, 50, samples=64)
    blobs = [{"object_id": 1, "object_key": "obj_1", "contour": contour, "area": 7800.0, "circularity": 0.95, "solidity": 0.99, "touches_border": False}]
    state = {
        "source_rgb": image,
        "blobs": blobs,
        "roi_offset": (0, 0),
        "active_2d_calibration": {
            "belt_plane": {
                "homography": [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]],
                "mm_per_px_x": 0.2,
                "mm_per_px_y": 0.2,
            }
        },
    }
    step = PipelineStepConfig(step_id="ellipse_fitting", algorithm_key="image_2d.measure.ellipse_fit", enabled=True, params={}, ui_state={})

    _, result = PipelineExecutor2D().run_step(step, image=image, existing=state)

    metrics_artifact = next(item for item in result.artifacts if item["artifact_id"] == "ellipse_metrics")
    entries = (metrics_artifact.get("metadata") or {}).get("entries") or []
    assert entries
    first = entries[0]
    assert first["equivalent_diameter_mm"] is not None
    assert first["major_axis_mm"] is not None
    assert first["minor_axis_mm"] is not None


def cv_points_circle(cx: int, cy: int, r: int, samples: int = 64) -> np.ndarray:
    pts = []
    for i in range(samples):
        t = 2.0 * np.pi * i / samples
        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t)
        pts.append([x, y])
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return arr
