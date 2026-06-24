from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.synthetic import SyntheticNoiseConfig, create_advanced_synthetic_25d_take


def _primary_object(payload: dict) -> dict:
    objects = payload.get("objects") or []
    return max(objects, key=lambda row: float(row.get("footprint_area_mm2") or 0.0), default={})


def _g(obj: dict, group: str, key: str) -> float:
    return float(((obj.get(group) or {}).get(key) or 0.0))


def test_advanced_generator_persists_contract_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = create_advanced_synthetic_25d_take(
        data_dir,
        object_type="truncated_sphere",
        seed=1001,
        noise=SyntheticNoiseConfig(gaussian_height_std_mm=0.2, missing_region_fraction=0.01),
    )
    assert take_id
    metadata = (take_dir / "metadata.json").read_text(encoding="utf-8")
    assert "synthetic_object_type" in metadata
    assert "generation_parameters" in metadata
    assert "expected_failure_modes" in metadata
    assert "expected_metric_family_reactions" in metadata


def test_metric_families_discriminate_good_vs_truncated_vs_scrap(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    noise = SyntheticNoiseConfig(gaussian_height_std_mm=0.25, missing_region_fraction=0.0)
    good_id, _ = create_advanced_synthetic_25d_take(data_dir, object_type="good_ball", seed=1101, noise=noise)
    trunc_id, _ = create_advanced_synthetic_25d_take(data_dir, object_type="truncated_sphere", seed=1102, noise=noise)
    scrap_id, _ = create_advanced_synthetic_25d_take(data_dir, object_type="elongated_scrap", seed=1103, noise=noise)

    good = _primary_object(run_ball_inspection_25d_flow(data_dir, take_id=good_id).result_payload)
    trunc = _primary_object(run_ball_inspection_25d_flow(data_dir, take_id=trunc_id).result_payload)
    scrap = _primary_object(run_ball_inspection_25d_flow(data_dir, take_id=scrap_id).result_payload)

    # Good sphere should remain better than truncated/scrap on coherence and deformation.
    assert _g(good, "sphere_consistency", "radial_height_rmse_mm") < _g(trunc, "sphere_consistency", "radial_height_rmse_mm")
    assert _g(trunc, "surface_geometry", "sphere_fit_rmse_mm") > _g(good, "surface_geometry", "sphere_fit_rmse_mm")
    assert _g(scrap, "footprint_geometry", "eccentricity") > _g(good, "footprint_geometry", "eccentricity")
    assert _g(scrap, "surface_geometry", "deformation_score") >= _g(good, "surface_geometry", "deformation_score")


def test_damage_family_reacts_to_chipped_and_flattened_modes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    noise = SyntheticNoiseConfig(gaussian_height_std_mm=0.30, missing_region_fraction=0.0)
    good_id, _ = create_advanced_synthetic_25d_take(data_dir, object_type="good_ball", seed=1201, noise=noise)
    chip_id, _ = create_advanced_synthetic_25d_take(data_dir, object_type="chipped_sphere", seed=1202, noise=noise)
    flat_id, _ = create_advanced_synthetic_25d_take(data_dir, object_type="flattened_ball", seed=1203, noise=noise)

    good = _primary_object(run_ball_inspection_25d_flow(data_dir, take_id=good_id).result_payload)
    chip = _primary_object(run_ball_inspection_25d_flow(data_dir, take_id=chip_id).result_payload)
    flat = _primary_object(run_ball_inspection_25d_flow(data_dir, take_id=flat_id).result_payload)

    assert _g(chip, "footprint_geometry", "radial_cv") > _g(good, "footprint_geometry", "radial_cv")
    assert _g(chip, "damage_metrics", "surface_discontinuity_score") >= _g(good, "damage_metrics", "surface_discontinuity_score")
    assert _g(flat, "sphere_consistency", "radial_height_rmse_mm") >= _g(good, "sphere_consistency", "radial_height_rmse_mm")


def test_validation_harness_exports_csv_and_analysis_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    cmd = [
        sys.executable,
        "scripts/run_advanced_25d_geometry_validation.py",
        "--data-dir",
        str(data_dir),
        "--samples-per-type",
        "1",
        "--seed",
        "5101",
        "--gaussian-noise",
        "0.2",
        "--missing-fraction",
        "0.0",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    reports = data_dir / "reports"
    csv_path = reports / "advanced_25d_geometry_validation.csv"
    analysis_path = reports / "advanced_25d_geometry_analysis.json"
    assert csv_path.is_file()
    assert analysis_path.is_file()
    csv_head = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "sample_id" in csv_head
    assert "synthetic_object_type" in csv_head
    assert "dominant_failure_family" in csv_head
    analysis = analysis_path.read_text(encoding="utf-8")
    assert '"feature_schema_version": "v1"' in analysis
    assert '"family_sensitivity_matrix"' in analysis
