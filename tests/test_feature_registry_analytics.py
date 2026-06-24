from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.ml.features import (
    FeatureDataset,
    FeatureRegistry,
    compute_distribution_by_object_type,
    export_feature_analytics_reports,
    run_feature_analytics,
)


def _fixture_dataset(tmp_path: Path) -> FeatureDataset:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "advanced_25d_geometry_validation.csv"
    analysis_path = reports / "advanced_25d_geometry_analysis.json"
    csv_path.write_text(
        "\n".join(
            [
                "sample_id,synthetic_object_type,seed,expected_failure_modes,expected_metric_family_reactions,predicted_class,predicted_subclass,classification_confidence,circularity,radial_cv,eccentricity,sphere_fit_rmse_mm,sphere_vs_ellipsoid_gain,deformation_score,radial_height_rmse_mm,surface_completeness_ratio,volume_deficit_ratio,surface_roughness_score,flat_region_ratio,surface_discontinuity_score,footprint_family_score,surface_family_score,consistency_family_score,damage_family_score,dominant_failure_family,secondary_failure_family,family_severity_ranking",
                's1,good_ball,1,[],{},ball,,0.9,0.96,0.02,0.12,0.3,0.01,0.05,0.5,0.99,0.12,0.03,0.04,0.10,0.10,0.20,0.10,0.10,none,none,"[]"',
                's2,good_ball,2,[],{},ball,,0.9,0.95,0.03,0.15,0.5,0.02,0.07,0.6,0.98,0.15,0.04,0.05,0.12,0.12,0.22,0.11,0.11,none,none,"[]"',
                's3,truncated_sphere,3,["truncation"],{"sphere_consistency":"FAIL"},non_ball,,0.6,0.88,0.11,0.52,3.0,0.11,0.41,5.9,0.79,0.44,0.08,0.25,0.58,0.36,0.41,0.71,0.31,sphere_consistency,damage_metrics,"[""sphere_consistency""]"',
                's4,elongated_scrap,4,["elongation"],{"footprint_geometry":"FAIL"},non_ball,,0.6,0.71,0.30,0.91,4.1,0.31,0.61,4.9,0.84,0.62,0.13,0.31,0.77,0.71,0.63,0.59,0.43,footprint_geometry,surface_geometry,"[""footprint_geometry""]"',
            ]
        ),
        encoding="utf-8",
    )
    analysis_path.write_text(
        json.dumps(
            {
                "feature_schema_version": "v1",
                "feature_groups": {
                    "footprint_geometry": ["circularity", "radial_cv", "eccentricity"],
                    "surface_geometry": ["sphere_fit_rmse_mm", "sphere_vs_ellipsoid_gain", "deformation_score"],
                    "sphere_consistency": ["radial_height_rmse_mm", "surface_completeness_ratio", "volume_deficit_ratio"],
                    "damage_metrics": ["surface_roughness_score", "flat_region_ratio", "surface_discontinuity_score"],
                },
                "feature_metadata": {
                    "circularity": {"group": "footprint_geometry", "display_name": "Circularity", "unit": "ratio", "higher_is_worse": False, "normalization_hint": "robust_zscore", "description": "c", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "radial_cv": {"group": "footprint_geometry", "display_name": "Radial CV", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "r", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "eccentricity": {"group": "footprint_geometry", "display_name": "Eccentricity", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "e", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "sphere_fit_rmse_mm": {"group": "surface_geometry", "display_name": "Sphere RMSE", "unit": "mm", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "s", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "sphere_vs_ellipsoid_gain": {"group": "surface_geometry", "display_name": "Sphere vs Ellipsoid", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "g", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "deformation_score": {"group": "surface_geometry", "display_name": "Deformation", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "d", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "radial_height_rmse_mm": {"group": "sphere_consistency", "display_name": "Radial Height RMSE", "unit": "mm", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "rh", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "surface_completeness_ratio": {"group": "sphere_consistency", "display_name": "Surface Completeness", "unit": "ratio", "higher_is_worse": False, "normalization_hint": "robust_zscore", "description": "sc", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "volume_deficit_ratio": {"group": "sphere_consistency", "display_name": "Volume Deficit", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "vd", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "surface_roughness_score": {"group": "damage_metrics", "display_name": "Roughness", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "sr", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "flat_region_ratio": {"group": "damage_metrics", "display_name": "Flat Region", "unit": "ratio", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "fr", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                    "surface_discontinuity_score": {"group": "damage_metrics", "display_name": "Discontinuity", "unit": "score", "higher_is_worse": True, "normalization_hint": "robust_zscore", "description": "sd", "ux_visibility": {"operations": False, "studio": True, "classifier_studio": True}},
                },
            }
        ),
        encoding="utf-8",
    )
    return FeatureDataset.from_advanced_validation_exports(csv_path=csv_path, analysis_json_path=analysis_path)


def test_feature_registry_validation_and_visibility(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    registry = FeatureRegistry.from_dataset(ds)
    meta_report = registry.validate_metadata()
    assert meta_report["ok"] is True
    compat = registry.validate_dataset_compatibility(ds)
    assert compat["ok"] is True
    studio_features = registry.visible_features(audience="studio")
    assert "radial_cv" in studio_features


def test_feature_analytics_reports_and_exports(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    registry = FeatureRegistry.from_dataset(ds)
    reports = run_feature_analytics(ds, registry)
    assert "features" in reports.quality_report
    assert "features" in reports.stability_report
    assert "correlation_matrix" in reports.correlation_report
    assert "features" in reports.readiness_report
    assert "operations" in reports.ux_summary
    out = tmp_path / "feature_reports"
    paths = export_feature_analytics_reports(reports, out)
    assert Path(paths["feature_quality_report"]).is_file()
    assert Path(paths["feature_stability_report"]).is_file()
    assert Path(paths["feature_correlation_report"]).is_file()
    assert Path(paths["feature_readiness_report"]).is_file()
    assert Path(paths["feature_drift_report"]).is_file()
    assert Path(paths["feature_ux_summary"]).is_file()


def test_distribution_and_drift(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    registry = FeatureRegistry.from_dataset(ds)
    dist = compute_distribution_by_object_type(ds, registry)
    assert "good_ball" in dist
    reports = run_feature_analytics(ds, registry)
    drift = run_feature_analytics(ds, registry, baseline_snapshot=reports.quality_report).drift_report
    assert "status" in drift
