from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.ml.features import FeatureDataset


def _write_fixture_exports(tmp_path: Path) -> tuple[Path, Path]:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "advanced_25d_geometry_validation.csv"
    analysis_path = reports / "advanced_25d_geometry_analysis.json"
    csv_path.write_text(
        "\n".join(
            [
                "sample_id,synthetic_object_type,seed,expected_failure_modes,expected_metric_family_reactions,predicted_class,predicted_subclass,classification_confidence,circularity,radial_cv,eccentricity,sphere_fit_rmse_mm,sphere_vs_ellipsoid_gain,deformation_score,radial_height_rmse_mm,surface_completeness_ratio,volume_deficit_ratio,surface_roughness_score,flat_region_ratio,surface_discontinuity_score,footprint_family_score,surface_family_score,consistency_family_score,damage_family_score,dominant_failure_family,secondary_failure_family,family_severity_ranking",
                's1,good_ball,1,[],{},ball,,0.9,0.95,0.02,0.1,0.4,0.01,0.1,0.5,0.99,0.12,0.03,0.04,0.10,0.10,0.20,0.10,0.10,none,none,"[]"',
                's2,truncated_sphere,2,["truncation"],{"sphere_consistency":"FAIL"},non_ball,,0.6,0.90,0.10,0.5,3.1,0.12,0.4,5.8,0.80,0.40,0.07,0.22,0.60,0.35,0.40,0.70,0.30,sphere_consistency,damage_metrics,"[""sphere_consistency"",""damage_metrics""]"',
            ]
        ),
        encoding="utf-8",
    )
    analysis_payload = {
        "feature_schema_version": "v1",
        "feature_groups": {
            "footprint_geometry": ["circularity", "radial_cv", "eccentricity"],
            "surface_geometry": ["sphere_fit_rmse_mm", "sphere_vs_ellipsoid_gain", "deformation_score"],
            "sphere_consistency": ["radial_height_rmse_mm", "surface_completeness_ratio", "volume_deficit_ratio"],
            "damage_metrics": ["surface_roughness_score", "flat_region_ratio", "surface_discontinuity_score"],
        },
        "feature_metadata": {
            "circularity": {"group": "footprint_geometry", "unit": "ratio", "higher_is_worse": False, "description": "c"},
            "radial_cv": {"group": "footprint_geometry", "unit": "ratio", "higher_is_worse": True, "description": "r"},
            "eccentricity": {"group": "footprint_geometry", "unit": "ratio", "higher_is_worse": True, "description": "e"},
            "sphere_fit_rmse_mm": {"group": "surface_geometry", "unit": "mm", "higher_is_worse": True, "description": "s"},
            "sphere_vs_ellipsoid_gain": {"group": "surface_geometry", "unit": "ratio", "higher_is_worse": True, "description": "g"},
            "deformation_score": {"group": "surface_geometry", "unit": "ratio", "higher_is_worse": True, "description": "d"},
            "radial_height_rmse_mm": {"group": "sphere_consistency", "unit": "mm", "higher_is_worse": True, "description": "h"},
            "surface_completeness_ratio": {"group": "sphere_consistency", "unit": "ratio", "higher_is_worse": False, "description": "sc"},
            "volume_deficit_ratio": {"group": "sphere_consistency", "unit": "ratio", "higher_is_worse": True, "description": "v"},
            "surface_roughness_score": {"group": "damage_metrics", "unit": "ratio", "higher_is_worse": True, "description": "sr"},
            "flat_region_ratio": {"group": "damage_metrics", "unit": "ratio", "higher_is_worse": True, "description": "fr"},
            "surface_discontinuity_score": {"group": "damage_metrics", "unit": "score", "higher_is_worse": True, "description": "sd"},
        },
    }
    analysis_path.write_text(json.dumps(analysis_payload), encoding="utf-8")
    return csv_path, analysis_path


def test_feature_dataset_ingestion_and_group_preservation(tmp_path: Path) -> None:
    csv_path, analysis_path = _write_fixture_exports(tmp_path)
    dataset = FeatureDataset.from_advanced_validation_exports(csv_path=csv_path, analysis_json_path=analysis_path)
    assert dataset.feature_schema.feature_schema_version == "v1"
    assert len(dataset.samples) == 2
    assert "footprint_geometry" in dataset.get_feature_groups()
    assert dataset.samples[0].metadata.get("synthetic_object_type") == "good_ball"
    assert dataset.samples[1].metadata.get("expected_failure_modes") == ["truncation"]
    matrix, names = dataset.get_feature_matrix()
    assert len(matrix) == 2
    assert len(names) == 12
    assert names[0] == "circularity"


def test_feature_dataset_validation_and_statistics(tmp_path: Path) -> None:
    csv_path, analysis_path = _write_fixture_exports(tmp_path)
    dataset = FeatureDataset.from_advanced_validation_exports(csv_path=csv_path, analysis_json_path=analysis_path)
    report = dataset.validate()
    assert report["ok"] is True
    stats = dataset.compute_statistics()
    assert "features" in stats and "groups" in stats
    assert "circularity" in stats["features"]
    assert "footprint_geometry" in stats["groups"]


def test_feature_dataset_persistence_and_reload(tmp_path: Path) -> None:
    csv_path, analysis_path = _write_fixture_exports(tmp_path)
    dataset = FeatureDataset.from_advanced_validation_exports(csv_path=csv_path, analysis_json_path=analysis_path)
    out = tmp_path / "feature_dataset.json"
    dataset.save_json(out)
    loaded = FeatureDataset.load_json(out)
    assert loaded.dataset_id == dataset.dataset_id
    assert loaded.get_feature_names() == dataset.get_feature_names()
    assert loaded.get_labels(field_name="synthetic_object_type") == ["good_ball", "truncated_sphere"]


def test_feature_dataset_group_selection_and_normalization(tmp_path: Path) -> None:
    csv_path, analysis_path = _write_fixture_exports(tmp_path)
    dataset = FeatureDataset.from_advanced_validation_exports(csv_path=csv_path, analysis_json_path=analysis_path)
    subset = dataset.select_feature_groups(["sphere_consistency"])
    names = subset.get_feature_names()
    assert names == ["radial_height_rmse_mm", "surface_completeness_ratio", "volume_deficit_ratio"]
    norm = subset.normalize(mode="zscore")
    matrix, _ = norm.get_feature_matrix()
    assert len(matrix) == 2
