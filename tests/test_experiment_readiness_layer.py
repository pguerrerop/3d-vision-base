from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.ml.experiments import (
    ExperimentConfig,
    LabelTaxonomy,
    create_split_set,
    load_split_manifest,
    run_baseline_experiment,
    save_split_manifest,
    validate_experiment_compatibility,
)
from vision_3d_acquisition.ml.features import FeatureDataset


def _fixture_dataset(tmp_path: Path) -> FeatureDataset:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "advanced_25d_geometry_validation.csv"
    analysis_path = reports / "advanced_25d_geometry_analysis.json"
    rows = [
        "sample_id,synthetic_object_type,seed,expected_failure_modes,expected_metric_family_reactions,predicted_class,predicted_subclass,classification_confidence,circularity,radial_cv,eccentricity,sphere_fit_rmse_mm,sphere_vs_ellipsoid_gain,deformation_score,radial_height_rmse_mm,surface_completeness_ratio,volume_deficit_ratio,surface_roughness_score,flat_region_ratio,surface_discontinuity_score,footprint_family_score,surface_family_score,consistency_family_score,damage_family_score,dominant_failure_family,secondary_failure_family,family_severity_ranking",
    ]
    for i in range(12):
        cls = "ball" if i % 2 == 0 else "non_ball"
        synth = "good_ball" if i % 3 == 0 else ("truncated_sphere" if i % 3 == 1 else "elongated_scrap")
        dom = "sphere_consistency" if synth == "truncated_sphere" else ("footprint_geometry" if synth == "elongated_scrap" else "none")
        rows.append(
            f's{i},{synth},{100+i},[],{{}},{cls},,0.8,0.9,0.0{i%4+1},0.{i%5+1},1.{i%3},0.0{i%2+1},0.2,1.{i%4},0.9,0.2,0.1,0.2,0.3,0.2,0.2,0.2,0.2,{dom},damage_metrics,"[]"'
        )
    csv_path.write_text("\n".join(rows), encoding="utf-8")
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
                "feature_metadata": {k: {"group": g, "unit": "ratio", "higher_is_worse": True, "description": k} for g, cols in {
                    "footprint_geometry": ["circularity", "radial_cv", "eccentricity"],
                    "surface_geometry": ["sphere_fit_rmse_mm", "sphere_vs_ellipsoid_gain", "deformation_score"],
                    "sphere_consistency": ["radial_height_rmse_mm", "surface_completeness_ratio", "volume_deficit_ratio"],
                    "damage_metrics": ["surface_roughness_score", "flat_region_ratio", "surface_discontinuity_score"],
                }.items() for k in cols},
            }
        ),
        encoding="utf-8",
    )
    return FeatureDataset.from_advanced_validation_exports(csv_path=csv_path, analysis_json_path=analysis_path, dataset_id="d_exp", name="exp")


def test_split_determinism_and_persistence(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    a = create_split_set(ds, strategy="stratified", seed=7)
    b = create_split_set(ds, strategy="stratified", seed=7)
    assert a.train_ids == b.train_ids
    manifest = tmp_path / "split_manifest.json"
    save_split_manifest(a, manifest)
    loaded = load_split_manifest(manifest)
    assert loaded.train_ids == a.train_ids
    assert loaded.split_strategy == "stratified"


def test_split_strategies_supported(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    by_obj = create_split_set(ds, strategy="by_synthetic_object_type", seed=11)
    by_fam = create_split_set(ds, strategy="by_failure_family", seed=11)
    assert len(by_obj.train_ids) + len(by_obj.validation_ids) + len(by_obj.test_ids) == len(ds.samples)
    assert len(by_fam.train_ids) + len(by_fam.validation_ids) + len(by_fam.test_ids) == len(ds.samples)


def test_compatibility_validation_and_label_taxonomy(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    cfg = ExperimentConfig(
        experiment_id="e1",
        name="baseline",
        feature_schema_version="v1",
        dataset_id="d_exp",
        split_config={"strategy": "stratified"},
        feature_selection={"include_groups": ["sphere_consistency", "damage_metrics"]},
        label_config={},
        normalization={"mode": "zscore"},
        evaluation={"backend": "majority_label_baseline"},
    )
    report = validate_experiment_compatibility(ds, cfg)
    assert report["ok"] is True
    taxonomy = LabelTaxonomy(
        classes=["ball", "non_ball"],
        aliases={"good_ball": "ball"},
        groups={"scrap": ["non_ball"]},
        target_mode="binary",
        positive_class="ball",
    )
    assert taxonomy.normalize_label("good_ball") == "ball"
    assert taxonomy.normalize_label("non_ball") == "non_ball"


def test_baseline_evaluation_flow_and_artifacts(tmp_path: Path) -> None:
    ds = _fixture_dataset(tmp_path)
    split = create_split_set(ds, strategy="stratified", seed=19)
    cfg = ExperimentConfig(
        experiment_id="ablation_damage_consistency",
        name="ablation",
        feature_schema_version="v1",
        dataset_id="d_exp",
        split_config={"strategy": "stratified", "seed": 19},
        feature_selection={"include_groups": ["sphere_consistency", "damage_metrics"]},
        label_config={},
        normalization={"mode": "minmax"},
        evaluation={"backend": "majority_label_baseline"},
    )
    taxonomy = LabelTaxonomy(classes=["ball", "non_ball"])
    result = run_baseline_experiment(dataset=ds, split_set=split, config=cfg, taxonomy=taxonomy, output_root=tmp_path / "ml" / "experiments")
    assert result["ok"] is True
    exp_dir = Path(result["experiment_dir"])
    assert (exp_dir / "config.json").is_file()
    assert (exp_dir / "split_manifest.json").is_file()
    assert (exp_dir / "evaluation.json").is_file()
    assert (exp_dir / "feature_stats.json").is_file()
    assert (exp_dir / "normalization_manifest.json").is_file()
