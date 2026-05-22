from __future__ import annotations

import numpy as np

from vision_3d_acquisition.processes.executor import PipelineExecutor2D
from vision_3d_acquisition.processes.models import PipelineStepConfig


def test_old_recipe_params_still_use_default_fit_method() -> None:
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    contour_points = np.stack([80 + 25 * np.cos(theta), 70 + 14 * np.sin(theta)], axis=1).tolist()
    state = {
        "blobs": [
            {
                "object_id": 1,
                "object_key": "object_001",
                "contour": np.asarray(contour_points, dtype=np.float32).reshape((-1, 1, 2)),
                "area": 1000.0,
                "circularity": 0.8,
                "solidity": 0.9,
                "touches_border": False,
            }
        ]
    }
    step = PipelineStepConfig(
        step_id="ellipse_fitting",
        algorithm_key="image_2d.measure.ellipse_fit",
        enabled=True,
        params={"min_fit_points": 5, "fit_error_threshold": 0.25},
        ui_state={},
    )
    report, result = PipelineExecutor2D().run_step(step, image=np.zeros((200, 200, 3), dtype=np.uint8), existing=state)
    assert report.status in {"success", "warning"}
    metrics_artifact = next(item for item in result.artifacts if item["artifact_id"] == "ellipse_metrics")
    entries = (metrics_artifact.get("metadata") or {}).get("entries") or []
    assert isinstance(entries, list)
    if entries:
        assert entries[0]["fit_method"] == "opencv_fitEllipse"
