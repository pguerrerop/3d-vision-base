from __future__ import annotations

import numpy as np
import pytest
import cv2

from vision_3d_acquisition.processes.executor import _ellipse_config_from_step_params, _fit_ellipse_with_method


def _synthetic_contour() -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x = 120.0 + 40.0 * np.cos(theta)
    y = 80.0 + 22.0 * np.sin(theta)
    pts = np.stack([x, y], axis=1).astype(np.float32)
    return pts.reshape((-1, 1, 2))


def test_ellipse_config_defaults_and_aliases() -> None:
    cfg = _ellipse_config_from_step_params({"min_fit_points": 11})
    assert cfg.fit_method == "opencv_fitEllipse"
    assert cfg.min_contour_points == 11
    assert cfg.ransac_iterations == 250
    assert cfg.refinement_method == "none"


def test_ellipse_config_validation_errors() -> None:
    with pytest.raises(ValueError):
        _ellipse_config_from_step_params({"min_contour_points": 3})
    with pytest.raises(ValueError):
        _ellipse_config_from_step_params({"ransac_min_inlier_ratio": 1.2})
    with pytest.raises(ValueError):
        _ellipse_config_from_step_params({"refinement_outlier_weight": 1.5})


def test_method_selection_routes_to_expected_fitter_shapes() -> None:
    contour = _synthetic_contour()
    rng = np.random.default_rng(7)
    methods = ["opencv_fitEllipse", "ransac_ellipse"]
    if hasattr(cv2, "fitEllipseDirect"):
        methods.append("opencv_fitEllipseDirect")
    if hasattr(cv2, "fitEllipseAMS"):
        methods.append("opencv_fitEllipseAMS")
    for method in methods:
        result = _fit_ellipse_with_method(
            contour,
            method=method,
            ransac_iterations=120,
            ransac_inlier_threshold_px=3.0,
            ransac_min_inlier_ratio=0.3,
            rng=rng,
        )
        ellipse = result["ellipse"]
        assert len(ellipse) == 3
        assert result["point_count"] >= 5
