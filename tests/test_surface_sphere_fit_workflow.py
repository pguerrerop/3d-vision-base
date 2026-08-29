from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import sensor_studio_cli as cli
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml.features.surface_sphere_fit_workflow import (
    BACKFILL_SIDECAR_NAME,
    backfill_take_feature,
    inspect_take_feature,
)
from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, save_heightmap_npz


def _write_old_style_result(data_dir: Path, take_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "metadata.json").write_text(json.dumps({"take_id": take_id, "created_at": "2026-05-28T12:00:00Z"}), encoding="utf-8")
    (take_dir / "READY").touch()
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-28T12:01:00Z",
                "status": "ok",
                "run_id": "run_old_1",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "feature_eccentricity": 0.2,
                        "bbox_px": [40, 40, 80, 80],
                        "contour_px": [[40, 40], [120, 40], [120, 120], [40, 120]],
                        "point_count": 120,
                    }
                ],
                "files": {"normalized_heightmap": "normalized_heightmap.npz"},
            }
        ),
        encoding="utf-8",
    )
    y, x = np.mgrid[0:160, 0:160]
    z = np.sqrt(np.maximum(0.0, 40.0**2 - (x - 80.0) ** 2 - (y - 80.0) ** 2)).astype(np.float32)
    frame = HeightmapFrame(
        z_mm=z,
        valid_mask=z > 0.0,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="belt_plane_normalized_mm",
    )
    save_heightmap_npz(frame, result_dir / "normalized_heightmap.npz")
    (result_dir / "DONE").touch()


def test_inspect_feature_reports_missing_on_old_style_result(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_old_sphere"
    _write_old_style_result(data_dir, take_id)

    summary = inspect_take_feature(processed_dir=data_dir / "processed", take_id=take_id)
    assert summary["object_count"] == 1
    assert summary["objects_with_feature_value"] == 0
    assert summary["likely_reason"] in {
        "old_run_before_feature_existed",
        "feature_not_emitted_by_pipeline",
    }


def test_backfill_writes_provenance_sidecar(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_old_sphere"
    _write_old_style_result(data_dir, take_id)

    result = backfill_take_feature(
        processed_dir=data_dir / "processed",
        take_id=take_id,
        dry_run=False,
    )
    sidecar_path = Path(result["sidecar_path"])
    assert sidecar_path.is_file()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["feature_key"] == "surface_sphere_fit_rmse_mm"
    assert payload["feature_algorithm_version"] == "surface_sphere_fit_v1"
    assert payload["source"] == "feature_backfill"
    assert payload["entries"]
    entry = payload["entries"][0]
    assert entry["source"] == "feature_backfill"
    assert entry["pipeline_id"] == "mining_steel_ball_classification_25d"
    assert entry["value"] is not None


def test_cli_25d_inspect_feature(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_old_sphere"
    _write_old_style_result(data_dir, take_id)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "25d",
            "inspect-feature",
            "--data-dir",
            str(data_dir),
            "--take-id",
            take_id,
            "--feature",
            "surface_sphere_fit_rmse_mm",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert "25D feature inspection" in out
    assert take_id in out
    assert code == 1


def test_cli_ml_compute_feature_backfill_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_old_sphere"
    _write_old_style_result(data_dir, take_id)
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="balls_scrap_2026_05_25_29_table_v1", name="balls", task_type="classification")
    service.upsert_take_metadata(
        take_id=take_id,
        dataset_id="d1",
        session_id="s1",
        updates={},
        source_metadata={"created_at": "2026-05-28T12:00:00Z"},
    )
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="balls_scrap_2026_05_25_29_table_v1", take_id=take_id)

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "compute-feature",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "balls_scrap_2026_05_25_29_table_v1",
            "--feature",
            "surface_sphere_fit_rmse_mm",
            "--mode",
            "backfill",
            "--dry-run",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "ML set feature compute" in out
    assert not (data_dir / "processed" / take_id / BACKFILL_SIDECAR_NAME).is_file()

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "compute-feature",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "balls_scrap_2026_05_25_29_table_v1",
            "--feature",
            "surface_sphere_fit_rmse_mm",
            "--mode",
            "backfill",
            "--apply",
        ],
    )
    code = cli.main()
    assert code == 0
    assert (data_dir / "processed" / take_id / BACKFILL_SIDECAR_NAME).is_file()
