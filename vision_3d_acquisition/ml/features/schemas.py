from __future__ import annotations

FEATURE_SCHEMA_VERSION = "1.0.0"

# Stable ordering for deterministic training/inference.
FEATURE_COLUMNS = [
    "feature_eccentricity",
    "feature_sphericity_3d",
    "feature_flatness",
    "feature_edge_roughness",
    "feature_volume_proxy_mm3",
    "height_max_mm",
    "height_p95_mm",
    "height_mean_mm",
    "footprint_radial_cv",
    "surface_sphere_fit_rmse_mm",
    "surface_sphere_fit_rmse_norm",
    "surface_sphere_fit_residual_p95_norm",
    "surface_sphere_fit_residual_mad_norm",
    "surface_sphere_radius_error_norm",
    "surface_visible_cap_fraction",
    "surface_volume_fill_ratio",
    "surface_sphere_fit_confidence",
    "surface_deformation_score",
    "damage_flat_region_ratio",
    "damage_surface_discontinuity_score",
]


import hashlib

def feature_ordering_hash(columns: list[str] | None = None) -> str:
    cols = columns or FEATURE_COLUMNS
    return hashlib.sha1("|".join(cols).encode("utf-8")).hexdigest()

def schema_fingerprint(version: str = FEATURE_SCHEMA_VERSION, columns: list[str] | None = None) -> str:
    cols = columns or FEATURE_COLUMNS
    return hashlib.sha1((version + "::" + "|".join(cols)).encode("utf-8")).hexdigest()
