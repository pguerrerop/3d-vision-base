from __future__ import annotations

import math

import numpy as np

from vision_3d_acquisition.api.feature_analytics import _extract_feature_values
from vision_3d_acquisition.api.feature_catalog import feature_definition_for_key
from vision_3d_acquisition.vision_core.geometry.sphere_consistency_features import (
    compute_sphere_consistency_features,
    compute_visible_cap_fraction,
    derive_sphere_consistency_from_object,
    resolve_reference_diameter_mm,
)
from vision_3d_acquisition.vision_core.geometry.surface_sphere_fit import fit_sphere_least_squares


def _sphere_points(*, radius: float = 20.0, center=(0.0, 0.0, 20.0), count: int = 120) -> np.ndarray:
    rng = np.random.default_rng(3)
    points: list[list[float]] = []
    while len(points) < count:
        vec = rng.normal(size=3)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            continue
        direction = vec / norm
        candidate = np.asarray(center, dtype=np.float64) + direction * radius
        if candidate[2] >= 10.0:
            points.append(candidate.tolist())
    return np.asarray(points, dtype=np.float64)


def test_resolve_reference_diameter_fallback_chain() -> None:
    assert resolve_reference_diameter_mm(diameter_selected_mm=50.0)[0] == 50.0
    assert resolve_reference_diameter_mm(fitted_radius_mm=25.0)[0] == 50.0
    assert resolve_reference_diameter_mm(equivalent_diameter_mm=40.0)[0] == 40.0
    assert resolve_reference_diameter_mm(dim_x_mm=30.0, dim_y_mm=45.0, dim_z_mm=20.0)[0] == 45.0
    assert resolve_reference_diameter_mm()[0] is None


def test_normalized_rmse_scales_with_diameter() -> None:
    points = _sphere_points()
    rng = np.random.default_rng(9)
    points = points + rng.normal(scale=0.35, size=points.shape)
    fit = fit_sphere_least_squares(points)
    small = compute_sphere_consistency_features(
        sphere_fit=fit,
        diameter_selected_mm=40.0,
        height_p95_mm=18.0,
        volume_proxy_mm3=12000.0,
    )
    large = compute_sphere_consistency_features(
        sphere_fit=fit,
        diameter_selected_mm=80.0,
        height_p95_mm=18.0,
        volume_proxy_mm3=12000.0,
    )
    assert small["surface_sphere_fit_rmse_norm"] is not None
    assert large["surface_sphere_fit_rmse_norm"] is not None
    assert large["surface_sphere_fit_rmse_norm"] < small["surface_sphere_fit_rmse_norm"]


def test_null_denominator_produces_null_not_zero() -> None:
    fit = {
        "valid": True,
        "radius_mm": None,
        "rmse_mm": 1.5,
        "center_mm": [0.0, 0.0, 20.0],
        "point_count": 20,
        "residuals_mm": np.array([0.1, -0.2, 0.3]),
    }
    out = compute_sphere_consistency_features(sphere_fit=fit)
    assert out["surface_sphere_fit_rmse_norm"] is None
    assert out["surface_sphere_radius_error_norm"] is None


def test_residual_p95_and_mad_norm() -> None:
    residuals = np.asarray([0.0, 0.5, -0.5, 1.0, -1.0, 4.0], dtype=np.float64)
    fit = {
        "valid": True,
        "radius_mm": 20.0,
        "rmse_mm": float(np.sqrt(np.mean(residuals**2))),
        "center_mm": [0.0, 0.0, 20.0],
        "point_count": residuals.size,
        "residuals_mm": residuals,
    }
    out = compute_sphere_consistency_features(
        sphere_fit=fit,
        diameter_selected_mm=40.0,
        height_p95_mm=15.0,
        volume_proxy_mm3=5000.0,
    )
    assert out["surface_sphere_fit_residual_p95_norm"] is not None
    assert out["surface_sphere_fit_residual_mad_norm"] is not None
    assert out["surface_sphere_fit_residual_p95_norm"] > out["surface_sphere_fit_residual_mad_norm"]


def test_visible_cap_fraction_clamped() -> None:
    clamped, raw = compute_visible_cap_fraction(height_p95_mm=30.0, fitted_radius_mm=10.0)
    assert clamped == 1.0
    assert raw == 1.5
    clamped_low, raw_low = compute_visible_cap_fraction(height_p95_mm=4.0, fitted_radius_mm=20.0)
    assert clamped_low == 0.1
    assert math.isclose(raw_low, 0.1)


def test_volume_fill_ratio_model_metadata() -> None:
    fit = fit_sphere_least_squares(_sphere_points())
    out = compute_sphere_consistency_features(
        sphere_fit=fit,
        diameter_selected_mm=40.0,
        height_p95_mm=35.0,
        volume_proxy_mm3=12000.0,
    )
    assert out["surface_volume_fill_ratio_model"] in {"sphere", "spherical_cap", "unavailable"}
    assert out["surface_volume_fill_ratio"] is None or out["surface_volume_fill_ratio"] >= 0.0


def test_derive_sphere_consistency_from_legacy_object() -> None:
    raw = {
        "surface_geometry": {
            "sphere_fit_rmse_mm": 2.5,
            "sphere_fit_radius_mm": 20.0,
            "sphere_fit_center_mm": [0.0, 0.0, 20.0],
            "sphere_fit_valid_point_count": 100,
        },
        "height_above_belt_mm": {"p95_height_mm": 18.0},
        "feature_volume_proxy_mm3": 9000.0,
        "diameter_mm": 40.0,
        "dimensions_mm": [38.0, 39.0, 18.0],
        "sphere_consistency": {},
    }
    derived = derive_sphere_consistency_from_object(raw)
    assert "surface_sphere_fit_rmse_norm" in derived
    features, sources = _extract_feature_values(raw)
    assert features["surface_sphere_fit_rmse_norm"] == derived["surface_sphere_fit_rmse_norm"]
    assert sources["surface_sphere_fit_rmse_norm"] == "derived"


def test_feature_catalog_sphere_consistency_group() -> None:
    rmse = feature_definition_for_key("surface_sphere_fit_rmse_mm")
    norm = feature_definition_for_key("surface_sphere_fit_rmse_norm")
    assert rmse.display_name == "Raw sphere-fit RMSE"
    assert norm.family_label == "Sphere consistency"
    assert norm.stable_schema is True
    assert feature_definition_for_key("surface_sphere_center_depth_ratio").diagnostic_only is True
    assert feature_definition_for_key("surface_sphere_fit_confidence").diagnostic_only is True
