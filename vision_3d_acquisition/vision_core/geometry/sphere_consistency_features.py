from __future__ import annotations

from typing import Any

import numpy as np

from vision_3d_acquisition.vision_core.geometry.surface_sphere_fit import MIN_SPHERE_FIT_POINTS

SPHERE_CONSISTENCY_ALGORITHM_VERSION = "sphere_consistency_v1"
MAD_SCALE = 1.4826


def resolve_reference_diameter_mm(
    *,
    diameter_selected_mm: float | None = None,
    fitted_radius_mm: float | None = None,
    equivalent_diameter_mm: float | None = None,
    dim_x_mm: float | None = None,
    dim_y_mm: float | None = None,
    dim_z_mm: float | None = None,
) -> tuple[float | None, str]:
    if diameter_selected_mm is not None and diameter_selected_mm > 0.0:
        return float(diameter_selected_mm), "diameter_selected_mm"
    if fitted_radius_mm is not None and fitted_radius_mm > 0.0:
        return float(2.0 * fitted_radius_mm), "fitted_sphere_diameter"
    if equivalent_diameter_mm is not None and equivalent_diameter_mm > 0.0:
        return float(equivalent_diameter_mm), "equivalent_diameter"
    dims = [float(v) for v in (dim_x_mm, dim_y_mm, dim_z_mm) if v is not None and float(v) > 0.0]
    if dims:
        return float(max(dims)), "max_extent_diameter"
    return None, "unavailable"


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def _residual_p95_mm(residuals_mm: np.ndarray | None) -> float | None:
    if residuals_mm is None or residuals_mm.size == 0:
        return None
    return float(np.percentile(np.abs(np.asarray(residuals_mm, dtype=np.float64)), 95))


def _residual_mad_mm(residuals_mm: np.ndarray | None) -> float | None:
    if residuals_mm is None or residuals_mm.size == 0:
        return None
    abs_dev = np.abs(np.asarray(residuals_mm, dtype=np.float64) - np.median(residuals_mm))
    mad = float(np.median(abs_dev))
    return float(MAD_SCALE * mad)


def spherical_cap_volume_mm3(*, radius_mm: float, cap_height_mm: float) -> float | None:
    if radius_mm <= 0.0 or cap_height_mm <= 0.0:
        return None
    h = min(float(cap_height_mm), float(2.0 * radius_mm))
    return float(np.pi * h * h * (3.0 * radius_mm - h) / 3.0)


def full_sphere_volume_mm3(*, diameter_mm: float) -> float | None:
    if diameter_mm <= 0.0:
        return None
    radius = diameter_mm / 2.0
    return float((4.0 / 3.0) * np.pi * radius**3)


def compute_visible_cap_fraction(
    *,
    height_p95_mm: float | None,
    fitted_radius_mm: float | None,
) -> tuple[float | None, float | None]:
    if height_p95_mm is None or fitted_radius_mm is None or fitted_radius_mm <= 0.0:
        return None, None
    denom = 2.0 * fitted_radius_mm
    raw = float(height_p95_mm / denom)
    clamped = float(max(0.0, min(1.0, raw)))
    return clamped, raw


def compute_volume_fill_ratio(
    *,
    volume_proxy_mm3: float | None,
    fitted_radius_mm: float | None,
    reference_diameter_mm: float | None,
    cap_height_mm: float | None,
    visible_cap_fraction: float | None,
) -> tuple[float | None, str]:
    if volume_proxy_mm3 is None or volume_proxy_mm3 <= 0.0:
        return None, "unavailable"
    radius = fitted_radius_mm if fitted_radius_mm is not None and fitted_radius_mm > 0.0 else (
        (reference_diameter_mm / 2.0) if reference_diameter_mm is not None and reference_diameter_mm > 0.0 else None
    )
    if radius is None or radius <= 0.0:
        return None, "unavailable"

    use_full_sphere = visible_cap_fraction is not None and visible_cap_fraction >= 0.85
    if use_full_sphere and reference_diameter_mm is not None and reference_diameter_mm > 0.0:
        expected = full_sphere_volume_mm3(diameter_mm=reference_diameter_mm)
        model = "sphere"
    elif cap_height_mm is not None and cap_height_mm > 0.0:
        expected = spherical_cap_volume_mm3(radius_mm=radius, cap_height_mm=cap_height_mm)
        model = "spherical_cap"
    elif reference_diameter_mm is not None and reference_diameter_mm > 0.0:
        expected = full_sphere_volume_mm3(diameter_mm=reference_diameter_mm)
        model = "sphere"
    else:
        return None, "unavailable"

    if expected is None or expected <= 0.0:
        return None, "unavailable"
    return float(volume_proxy_mm3 / expected), model


def compute_center_depth_ratio(
    *,
    fitted_center_z_mm: float | None,
    fitted_radius_mm: float | None,
    reference_diameter_mm: float | None,
    visible_cap_fraction: float | None,
    min_cap_fraction: float = 0.35,
) -> tuple[float | None, str, str | None]:
    if (
        fitted_center_z_mm is None
        or fitted_radius_mm is None
        or fitted_radius_mm <= 0.0
        or reference_diameter_mm is None
        or reference_diameter_mm <= 0.0
    ):
        return None, "unavailable", "Missing fitted center, radius, or reference diameter."
    if visible_cap_fraction is None or visible_cap_fraction < min_cap_fraction:
        return (
            None,
            "unreliable_geometry",
            "Visible cap coverage is too low to infer a reliable expected sphere center depth.",
        )
    expected_center_z = fitted_radius_mm
    ratio = _safe_ratio(abs(fitted_center_z_mm - expected_center_z), reference_diameter_mm)
    if ratio is None:
        return None, "unavailable", "Could not normalize center depth ratio."
    return ratio, "ok", None


def compute_sphere_fit_confidence(
    *,
    sphere_fit_valid: bool,
    point_count: int,
    visible_cap_fraction: float | None,
    rmse_norm: float | None,
    mad_norm: float | None,
    radius_error_norm: float | None,
    segmentation_coverage: float | None,
    border_clipped: bool | None = None,
) -> tuple[float | None, dict[str, Any]]:
    if not sphere_fit_valid:
        return None, {"reason": "invalid_sphere_fit"}

    components: dict[str, float] = {}
    if point_count >= MIN_SPHERE_FIT_POINTS:
        components["point_count"] = 1.0
    elif point_count >= max(4, MIN_SPHERE_FIT_POINTS // 2):
        components["point_count"] = 0.5
    else:
        components["point_count"] = 0.0

    if visible_cap_fraction is None:
        components["visible_cap"] = 0.0
    elif visible_cap_fraction >= 0.55:
        components["visible_cap"] = 1.0
    elif visible_cap_fraction >= 0.30:
        components["visible_cap"] = 0.6
    else:
        components["visible_cap"] = 0.2

    residual_score = 0.0
    if rmse_norm is not None:
        residual_score += 0.5 * float(max(0.0, min(1.0, 1.0 - (rmse_norm / 0.25))))
    if mad_norm is not None:
        residual_score += 0.5 * float(max(0.0, min(1.0, 1.0 - (mad_norm / 0.20))))
    components["normalized_residuals"] = residual_score

    if radius_error_norm is None:
        components["radius_plausibility"] = 0.0
    elif radius_error_norm <= 0.15:
        components["radius_plausibility"] = 1.0
    elif radius_error_norm <= 0.35:
        components["radius_plausibility"] = 0.5
    else:
        components["radius_plausibility"] = 0.0

    if segmentation_coverage is None:
        components["segmentation_quality"] = 0.5
    elif segmentation_coverage >= 0.55:
        components["segmentation_quality"] = 1.0
    elif segmentation_coverage >= 0.35:
        components["segmentation_quality"] = 0.6
    else:
        components["segmentation_quality"] = 0.2

    components["border_clip_penalty"] = 0.0 if border_clipped else 1.0

    weights = {
        "point_count": 0.15,
        "visible_cap": 0.20,
        "normalized_residuals": 0.25,
        "radius_plausibility": 0.20,
        "segmentation_quality": 0.15,
        "border_clip_penalty": 0.05,
    }
    score = sum(float(components[key]) * weights[key] for key in weights)
    metadata = {
        "algorithm_version": SPHERE_CONSISTENCY_ALGORITHM_VERSION,
        "components": {key: round(float(value), 4) for key, value in components.items()},
        "weights": weights,
        "formula": "Weighted sum of point count, visible cap, normalized residuals, radius plausibility, segmentation quality, and border-clip penalty.",
    }
    return round(float(max(0.0, min(1.0, score))), 6), metadata


def compute_sphere_consistency_features(
    *,
    sphere_fit: dict[str, Any],
    ellipsoid_fit: dict[str, Any] | None = None,
    diameter_selected_mm: float | None = None,
    equivalent_diameter_mm: float | None = None,
    dim_x_mm: float | None = None,
    dim_y_mm: float | None = None,
    dim_z_mm: float | None = None,
    height_p95_mm: float | None = None,
    volume_proxy_mm3: float | None = None,
    segmentation_coverage: float | None = None,
    border_clipped: bool | None = None,
) -> dict[str, Any]:
    valid = bool(sphere_fit.get("valid"))
    fitted_radius_mm = float(sphere_fit["radius_mm"]) if valid and sphere_fit.get("radius_mm") is not None else None
    center_mm = sphere_fit.get("center_mm") if valid else None
    fitted_center_z_mm = float(center_mm[2]) if isinstance(center_mm, (list, tuple)) and len(center_mm) >= 3 else None
    residuals = sphere_fit.get("residuals_mm") if valid else None
    residuals_arr = np.asarray(residuals, dtype=np.float64) if residuals is not None else None
    rmse_mm = float(sphere_fit["rmse_mm"]) if valid and sphere_fit.get("rmse_mm") is not None else None
    point_count = int(sphere_fit.get("point_count") or (residuals_arr.shape[0] if residuals_arr is not None else 0))

    reference_diameter_mm, diameter_source = resolve_reference_diameter_mm(
        diameter_selected_mm=diameter_selected_mm,
        fitted_radius_mm=fitted_radius_mm,
        equivalent_diameter_mm=equivalent_diameter_mm,
        dim_x_mm=dim_x_mm,
        dim_y_mm=dim_y_mm,
        dim_z_mm=dim_z_mm,
    )

    rmse_norm = _safe_ratio(rmse_mm, reference_diameter_mm)
    p95_mm = _residual_p95_mm(residuals_arr)
    mad_mm = _residual_mad_mm(residuals_arr)
    p95_norm = _safe_ratio(p95_mm, reference_diameter_mm)
    mad_norm = _safe_ratio(mad_mm, reference_diameter_mm)

    radius_error_norm = None
    if fitted_radius_mm is not None and reference_diameter_mm is not None and reference_diameter_mm > 0.0:
        radius_error_norm = _safe_ratio(abs(2.0 * fitted_radius_mm - reference_diameter_mm), reference_diameter_mm)

    visible_cap_fraction, visible_cap_fraction_raw = compute_visible_cap_fraction(
        height_p95_mm=height_p95_mm,
        fitted_radius_mm=fitted_radius_mm,
    )
    center_depth_ratio, center_depth_status, center_depth_warning = compute_center_depth_ratio(
        fitted_center_z_mm=fitted_center_z_mm,
        fitted_radius_mm=fitted_radius_mm,
        reference_diameter_mm=reference_diameter_mm,
        visible_cap_fraction=visible_cap_fraction,
    )
    volume_fill_ratio, volume_fill_model = compute_volume_fill_ratio(
        volume_proxy_mm3=volume_proxy_mm3,
        fitted_radius_mm=fitted_radius_mm,
        reference_diameter_mm=reference_diameter_mm,
        cap_height_mm=height_p95_mm,
        visible_cap_fraction=visible_cap_fraction,
    )

    sphere_rmse = rmse_mm
    ellipsoid_rmse = None
    ellipsoid_valid = isinstance(ellipsoid_fit, dict) and bool(ellipsoid_fit.get("valid"))
    if ellipsoid_valid:
        ellipsoid_rmse = float(ellipsoid_fit.get("rmse") or 0.0)
    sphere_vs_ellipsoid_gain = None
    if sphere_rmse is not None and ellipsoid_rmse is not None and sphere_rmse > 1e-9:
        sphere_vs_ellipsoid_gain = float(max(0.0, (sphere_rmse - ellipsoid_rmse) / sphere_rmse))

    confidence, confidence_metadata = compute_sphere_fit_confidence(
        sphere_fit_valid=valid,
        point_count=point_count,
        visible_cap_fraction=visible_cap_fraction,
        rmse_norm=rmse_norm,
        mad_norm=mad_norm,
        radius_error_norm=radius_error_norm,
        segmentation_coverage=segmentation_coverage,
        border_clipped=border_clipped,
    )

    return {
        "surface_sphere_fit_rmse_norm": round(rmse_norm, 6) if rmse_norm is not None else None,
        "surface_sphere_fit_residual_p95_norm": round(p95_norm, 6) if p95_norm is not None else None,
        "surface_sphere_fit_residual_mad_norm": round(mad_norm, 6) if mad_norm is not None else None,
        "surface_sphere_radius_error_norm": round(radius_error_norm, 6) if radius_error_norm is not None else None,
        "surface_sphere_center_depth_ratio": round(center_depth_ratio, 6) if center_depth_ratio is not None else None,
        "surface_sphere_center_depth_ratio_status": center_depth_status,
        "surface_sphere_center_depth_ratio_warning": center_depth_warning,
        "surface_visible_cap_fraction": round(visible_cap_fraction, 6) if visible_cap_fraction is not None else None,
        "surface_visible_cap_fraction_raw": round(visible_cap_fraction_raw, 6) if visible_cap_fraction_raw is not None else None,
        "surface_volume_fill_ratio": round(volume_fill_ratio, 6) if volume_fill_ratio is not None else None,
        "surface_volume_fill_ratio_model": volume_fill_model,
        "surface_sphere_vs_ellipsoid_gain": round(sphere_vs_ellipsoid_gain, 6) if sphere_vs_ellipsoid_gain is not None else None,
        "surface_sphere_vs_ellipsoid_gain_available": bool(ellipsoid_valid),
        "surface_sphere_fit_confidence": confidence,
        "surface_sphere_fit_confidence_metadata": confidence_metadata if confidence is not None else None,
        "surface_sphere_reference_diameter_mm": round(reference_diameter_mm, 4) if reference_diameter_mm is not None else None,
        "surface_sphere_reference_diameter_source": diameter_source,
        "algorithm_version": SPHERE_CONSISTENCY_ALGORITHM_VERSION,
    }


DERIVABLE_SPHERE_CONSISTENCY_KEYS = (
    "surface_sphere_fit_rmse_norm",
    "surface_sphere_fit_residual_p95_norm",
    "surface_sphere_fit_residual_mad_norm",
    "surface_sphere_radius_error_norm",
    "surface_visible_cap_fraction",
    "surface_volume_fill_ratio",
    "surface_sphere_vs_ellipsoid_gain",
    "surface_sphere_fit_confidence",
)


def derive_sphere_consistency_from_object(raw_object: dict[str, Any]) -> dict[str, float]:
    surface = raw_object.get("surface_geometry") if isinstance(raw_object.get("surface_geometry"), dict) else {}
    consistency = raw_object.get("sphere_consistency") if isinstance(raw_object.get("sphere_consistency"), dict) else {}
    hab = raw_object.get("height_above_belt_mm") if isinstance(raw_object.get("height_above_belt_mm"), dict) else {}
    provenance = raw_object.get("measurement_provenance") if isinstance(raw_object.get("measurement_provenance"), dict) else {}
    dims = raw_object.get("dimensions_mm") if isinstance(raw_object.get("dimensions_mm"), (list, tuple)) else None

    missing_keys = [key for key in DERIVABLE_SPHERE_CONSISTENCY_KEYS if consistency.get(key) is None]
    if not missing_keys:
        return {}

    sphere_fit = {
        "valid": surface.get("sphere_fit_rmse_mm") is not None and surface.get("sphere_fit_radius_mm") is not None,
        "radius_mm": surface.get("sphere_fit_radius_mm"),
        "center_mm": surface.get("sphere_fit_center_mm"),
        "rmse_mm": surface.get("sphere_fit_rmse_mm"),
        "point_count": surface.get("sphere_fit_valid_point_count"),
        "residuals_mm": None,
    }
    if not sphere_fit["valid"]:
        return {}

    footprint_area = raw_object.get("footprint_area_mm2")
    equivalent_diameter = None
    try:
        if footprint_area is not None and float(footprint_area) > 0.0:
            equivalent_diameter = float(np.sqrt((4.0 * float(footprint_area)) / np.pi))
    except (TypeError, ValueError):
        equivalent_diameter = None

    computed = compute_sphere_consistency_features(
        sphere_fit=sphere_fit,
        ellipsoid_fit={"valid": surface.get("ellipsoid_fit_rmse_mm") is not None, "rmse": surface.get("ellipsoid_fit_rmse_mm")}
        if surface.get("ellipsoid_fit_rmse_mm") is not None
        else None,
        diameter_selected_mm=_maybe_float(raw_object.get("diameter_selected_mm") or raw_object.get("diameter_mm") or raw_object.get("diameter_estimate_mm")),
        equivalent_diameter_mm=equivalent_diameter,
        dim_x_mm=_maybe_float(dims[0] if isinstance(dims, (list, tuple)) and len(dims) >= 1 else None),
        dim_y_mm=_maybe_float(dims[1] if isinstance(dims, (list, tuple)) and len(dims) >= 2 else None),
        dim_z_mm=_maybe_float(dims[2] if isinstance(dims, (list, tuple)) and len(dims) >= 3 else None),
        height_p95_mm=_maybe_float(hab.get("p95_height_mm")),
        volume_proxy_mm3=_maybe_float(raw_object.get("feature_volume_proxy_mm3")),
        segmentation_coverage=_maybe_float(provenance.get("segmentation_coverage")),
        border_clipped=None,
    )
    out: dict[str, float] = {}
    for key in missing_keys:
        value = computed.get(key)
        if value is not None:
            out[key] = float(value)
    return out


def _maybe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out
