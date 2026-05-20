from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import open3d as o3d
import pytest

from vision_3d_acquisition.calibration.calibration_models import ObjectFilterConfig, PlaneCalibration, SystemCalibration
from vision_3d_acquisition.contracts.result import ProcessingResult
from vision_3d_acquisition.processing.clustering import cluster_foreground
from vision_3d_acquisition.processing.pipeline import ProcessingConfig, process_take
from vision_3d_acquisition.processing.pointcloud_io import save_point_cloud_npz
from vision_3d_acquisition.processing.plane_segmentation import segment_dominant_plane
from vision_3d_acquisition.utils.time import utc_now_iso
from scripts.run_ball_inspection import resolve_take


REPO_ROOT = Path(__file__).resolve().parents[1]


def runtime_default_calibration(reference_take_id: str) -> SystemCalibration:
    return SystemCalibration(
        calibration_id="runtime_default_test",
        created_at="2026-05-16T00:00:00Z",
        reference_take_id=reference_take_id,
        planes=[
            PlaneCalibration(
                plane_id="belt",
                label="belt",
                plane_model=(0.0, 0.0, 1.0, 0.0),
                point_count=81,
                bbox_min_mm=(-20.0, -20.0, -1.0),
                bbox_max_mm=(20.0, 20.0, 1.0),
                extent_mm=(40.0, 40.0, 2.0),
                avg_z_mm=0.0,
                normal=(0.0, 0.0, 1.0),
                roi_polygon_xy_mm=[(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)],
            )
        ],
        object_filter=ObjectFilterConfig(min_height_above_belt_mm=1.0, max_height_above_belt_mm=20.0),
    )


def write_synthetic_ply(path: Path) -> None:
    points: list[tuple[float, float, float]] = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            points.append((float(x), float(y), 0.0))
    for base_x in (-12.0, 12.0):
        for dx in (0.0, 0.6, 1.2):
            for dy in (0.0, 0.6, 1.2):
                for dz in (0.0, 0.5):
                    points.append((base_x + dx, dy, 3.0 + dz))
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points)}",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                *[f"{x} {y} {z}" for x, y, z in points],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_take(data_dir: Path) -> tuple[str, Path]:
    take_id = "real_take_001"
    return write_named_take(data_dir, take_id, created_at=utc_now_iso())


def write_named_take(data_dir: Path, take_id: str, *, created_at: str) -> tuple[str, Path]:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True)
    write_synthetic_ply(take_dir / "point_cloud.ply")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "test",
                "mode": "offline",
                "created_at": created_at,
                "frame_count": 1,
                "files": {"point_cloud": "point_cloud.ply"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (take_dir / "READY").touch()
    return take_id, take_dir


def write_take_with_npz(data_dir: Path) -> tuple[str, Path]:
    take_id, take_dir = write_take(data_dir)
    pcd = o3d.io.read_point_cloud(str(take_dir / "point_cloud.ply"))
    save_point_cloud_npz(pcd, take_dir / "point_cloud.npz")
    metadata = json.loads((take_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["files"]["point_cloud_npz"] = "point_cloud.npz"
    (take_dir / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return take_id, take_dir


def test_run_ball_inspection_take_resolution_hierarchy(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_named_take(data_dir, "2026-05-17T040000_001", created_at="2026-05-17T04:00:00Z")
    latest_id, _ = write_named_take(data_dir, "2026-05-17T041300_001", created_at="2026-05-17T04:13:00Z")
    (data_dir / "processed" / latest_id).mkdir(parents=True)
    (data_dir / "processed" / latest_id / "DONE").touch()
    unprocessed_id, _ = write_named_take(data_dir, "2026-05-17T040500_001", created_at="2026-05-17T04:05:00Z")

    explicit = resolve_take(data_dir, "2026-05-17T040000_001")
    unprocessed = resolve_take(data_dir, None)
    (data_dir / "processed" / unprocessed.take_id).mkdir(parents=True)
    (data_dir / "processed" / unprocessed.take_id / "DONE").touch()

    assert explicit.take_id == "2026-05-17T040000_001"
    assert explicit.source == "explicit"
    assert unprocessed.take_id == unprocessed_id
    assert unprocessed.source == "latest_unprocessed"
    (data_dir / "processed" / "2026-05-17T040000_001").mkdir(parents=True)
    (data_dir / "processed" / "2026-05-17T040000_001" / "DONE").touch()
    fallback = resolve_take(data_dir, None)
    assert fallback.take_id == latest_id
    assert fallback.source == "latest"


def test_plane_segmentation_returns_plane_and_foreground(tmp_path: Path) -> None:
    ply = tmp_path / "synthetic.ply"
    write_synthetic_ply(ply)
    pcd = o3d.io.read_point_cloud(str(ply))

    result = segment_dominant_plane(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=200)

    assert result["plane_model"] is not None
    assert len(result["plane_cloud"].points) >= 70
    assert len(result["foreground_cloud"].points) >= 20


def test_clustering_finds_foreground_clusters(tmp_path: Path) -> None:
    ply = tmp_path / "synthetic.ply"
    write_synthetic_ply(ply)
    pcd = o3d.io.read_point_cloud(str(ply))
    segmented = segment_dominant_plane(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=200)

    clusters = cluster_foreground(
        segmented["foreground_cloud"],
        dbscan_eps=1.6,
        dbscan_min_points=4,
        min_cluster_points=6,
    )

    assert len(clusters) == 2
    assert all(len(cluster["point_indices"]) >= 6 for cluster in clusters)


def test_real_pipeline_generates_valid_result_and_debug_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = write_take(data_dir)
    output_dir = data_dir / "processed" / take_id

    result = process_take(
        take_dir,
        output_dir,
        ProcessingConfig(
            voxel_size=0.0,
            enable_outlier_removal=True,
            outlier_nb_neighbors=5,
            outlier_std_ratio=10.0,
            plane_distance_threshold=0.2,
            plane_num_iterations=200,
            dbscan_eps=1.6,
            dbscan_min_points=4,
            min_cluster_points=6,
        ),
    )

    validated = ProcessingResult.model_validate_json((output_dir / "result.json").read_text(encoding="utf-8"))
    assert validated.take_id == result.take_id == take_id
    assert validated.processing_mode == "real"
    assert validated.algorithm_stage == "segmentation"
    assert validated.summary.decision == "review"
    assert validated.summary.confidence is None
    assert validated.profiling is not None
    assert validated.profiling.total_ms >= validated.profiling.production_ms
    assert any(stage.name == "dbscan_clustering" for stage in validated.profiling.stages)
    assert len(validated.objects) == 2
    assert all(item.class_name == "unknown" and item.confidence is None for item in validated.objects)
    for filename in (
        "input_point_cloud_preview.png",
        "debug_plane_segmentation.png",
        "debug_foreground.png",
        "debug_clusters.png",
    ):
        assert (output_dir / filename).is_file()


def test_process_latest_real_creates_done(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = tmp_path / "processing.yaml"
    config.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "outlier_nb_neighbors: 5",
                "outlier_std_ratio: 10.0",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    take_id, _ = write_take(data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "process_latest_real.py"),
            "--data-dir",
            str(data_dir),
            "--config",
            str(config),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    processed_dir = data_dir / "processed" / take_id
    assert (processed_dir / "DONE").is_file()
    result = json.loads((processed_dir / "result.json").read_text(encoding="utf-8"))
    assert result["processing_mode"] == "real"
    assert result["algorithm_stage"] == "segmentation"
    assert result["processing_engine"] == "legacy"
    assert result["plane_mode"] == "auto"
    assert result["poc_summary"]["calibration"]["display"] == "NONE"
    assert result["poc_summary"]["calibration"]["status"] == "automatic plane estimation only"
    assert result["profiling"]["stages"]
    assert "event/state_update" in {stage["name"] for stage in result["profiling"]["stages"]}
    assert result["files"]["debug_clusters"] == "debug_clusters.png"
    assert (data_dir / "state" / "latest.json").is_file()
    assert (data_dir / "events" / "events.jsonl").is_file()


def test_run_ball_inspection_defaults_to_latest_ready_and_runtime_calibration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = tmp_path / "processing.yaml"
    config.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "enable_outlier_removal: false",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    older_id, _ = write_named_take(data_dir, "2026-05-17T040000_001", created_at="2026-05-17T04:00:00Z")
    take_id, _ = write_named_take(data_dir, "2026-05-17T041300_001", created_at="2026-05-17T04:13:00Z")
    calibration_path = tmp_path / "config" / "calibrations" / "runtime_default_test.json"
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text(
        runtime_default_calibration(take_id).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "runtime.json").write_text(
        json.dumps(
            {
                "default_calibration_file": "config/calibrations/runtime_default_test.json",
                "require_calibration_for_demo": True,
                "allow_auto_plane_without_calibration": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_ball_inspection.py"),
            "--data-dir",
            str(data_dir),
            "--config",
            str(config),
            "--skip-debug-images",
            "--profile",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Selected latest take:" in completed.stdout
    assert take_id in completed.stdout
    assert str(Path("config/calibrations/runtime_default_test.json")) in completed.stdout
    assert "Engine:\nnative" in completed.stdout
    assert "Processing..." in completed.stdout
    assert f"processed_take: {take_id}" in completed.stdout
    assert not (data_dir / "processed" / older_id / "DONE").exists()
    result = json.loads((data_dir / "processed" / take_id / "result.json").read_text(encoding="utf-8"))
    assert result["processing_engine"] == "native"
    assert result["calibration_active"] is True
    assert result["calibration_resolution_source"] == "runtime_default"


def test_process_latest_real_require_calibration_fails_without_calibration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = write_take(data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "process_latest_real.py"),
            "--data-dir",
            str(data_dir),
            "--require-calibration",
            "--non-interactive",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "calibration is required" in completed.stderr
    assert not (data_dir / "processed" / take_id / "DONE").exists()


def test_process_latest_real_uses_runtime_default_calibration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = write_take(data_dir)
    calibration_path = tmp_path / "config" / "calibrations" / "runtime_default_test.json"
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text(
        runtime_default_calibration(take_id).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "runtime.json").write_text(
        json.dumps(
            {
                "default_calibration_file": "config/calibrations/runtime_default_test.json",
                "require_calibration_for_demo": True,
                "allow_auto_plane_without_calibration": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "process_latest_real.py"),
            "--data-dir",
            str(data_dir),
            "--non-interactive",
            "--skip-debug-images",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads((data_dir / "processed" / take_id / "result.json").read_text(encoding="utf-8"))
    assert result["calibration_active"] is True
    assert result["calibration_resolution_source"] == "runtime_default"
    assert result["calibration_id"] == "runtime_default_test"
    assert result["poc_summary"]["calibration"]["resolution_source"] == "runtime_default"


def test_process_latest_real_native_engine_creates_compatible_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = tmp_path / "processing.yaml"
    config.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "enable_outlier_removal: false",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    take_id, _ = write_take(data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "process_latest_real.py"),
            "--data-dir",
            str(data_dir),
            "--config",
            str(config),
            "--engine",
            "native",
            "--skip-debug-images",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    processed_dir = data_dir / "processed" / take_id
    result = json.loads((processed_dir / "result.json").read_text(encoding="utf-8"))
    validated = ProcessingResult.model_validate(result)
    assert (processed_dir / "DONE").is_file()
    assert validated.take_id == take_id
    assert validated.processing_mode == "real"
    assert validated.algorithm_stage == "classification"
    assert validated.profiling is not None
    assert validated.timing_ms.total >= 0
    assert result["files"]["debug_clusters"] is None
    assert (data_dir / "state" / "latest.json").is_file()
    assert (data_dir / "state" / "runtime.json").is_file()
    assert (data_dir / "events" / "events.jsonl").is_file()


def test_process_latest_real_legacy_and_native_match_shared_fixture_fields(tmp_path: Path) -> None:
    config = tmp_path / "processing.yaml"
    config.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "enable_outlier_removal: false",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_data = tmp_path / "legacy_data"
    native_data = tmp_path / "native_data"
    take_id, _ = write_take(legacy_data)
    write_take(native_data)

    for engine, data_dir in (("legacy", legacy_data), ("native", native_data)):
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "process_latest_real.py"),
                "--data-dir",
                str(data_dir),
                "--config",
                str(config),
                "--engine",
                engine,
                "--skip-debug-images",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    legacy = ProcessingResult.model_validate_json(
        (legacy_data / "processed" / take_id / "result.json").read_text(encoding="utf-8")
    )
    native = ProcessingResult.model_validate_json(
        (native_data / "processed" / take_id / "result.json").read_text(encoding="utf-8")
    )

    assert native.take_id == legacy.take_id
    assert native.processing_mode == legacy.processing_mode == "real"
    assert native.input_stats is not None
    assert legacy.input_stats is not None
    assert native.input_stats.point_count == legacy.input_stats.point_count
    assert native.files.point_cloud == legacy.files.point_cloud
    assert len(native.objects) == len(legacy.objects)
    assert native.plane_model is not None
    assert legacy.plane_model is not None
    for native_value, legacy_value in zip(native.plane_model, legacy.plane_model, strict=True):
        assert native_value == pytest.approx(legacy_value, abs=1e-6)
    assert native.profiling is not None
    assert legacy.profiling is not None
    assert native.timing_ms.total >= 0
    assert legacy.timing_ms.total >= 0


def test_real_pipeline_uses_npz_when_available(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = write_take_with_npz(data_dir)
    output_dir = data_dir / "processed" / take_id

    result = process_take(
        take_dir,
        output_dir,
        ProcessingConfig(
            voxel_size=0.0,
            enable_outlier_removal=False,
            plane_distance_threshold=0.2,
            plane_num_iterations=200,
            dbscan_eps=1.6,
            dbscan_min_points=4,
            min_cluster_points=6,
        ),
        skip_debug_images=True,
    )

    assert result.profiling is not None
    assert result.files.point_cloud_npz == "point_cloud.npz"
    assert "load_point_cloud_fast" in {stage.name for stage in result.profiling.stages}
    assert "load_point_cloud" not in {stage.name for stage in result.profiling.stages}


def test_real_pipeline_can_force_ply_fallback_when_npz_exists(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = write_take_with_npz(data_dir)
    output_dir = data_dir / "processed" / take_id

    result = process_take(
        take_dir,
        output_dir,
        ProcessingConfig(
            voxel_size=0.0,
            enable_outlier_removal=False,
            plane_distance_threshold=0.2,
            plane_num_iterations=200,
            dbscan_eps=1.6,
            dbscan_min_points=4,
            min_cluster_points=6,
        ),
        skip_debug_images=True,
        prefer_fast_cloud=False,
    )

    assert result.profiling is not None
    assert "load_point_cloud" in {stage.name for stage in result.profiling.stages}
    assert "load_point_cloud_fast" not in {stage.name for stage in result.profiling.stages}


def test_process_latest_real_skip_debug_images_avoids_png_generation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = tmp_path / "processing.yaml"
    config.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "outlier_nb_neighbors: 5",
                "outlier_std_ratio: 10.0",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    take_id, _ = write_take(data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "process_latest_real.py"),
            "--data-dir",
            str(data_dir),
            "--config",
            str(config),
            "--skip-debug-images",
            "--profile",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Timing profile" in completed.stdout
    processed_dir = data_dir / "processed" / take_id
    result = json.loads((processed_dir / "result.json").read_text(encoding="utf-8"))
    assert result["files"]["input_preview"] is None
    assert result["files"]["debug_clusters"] is None
    assert not (processed_dir / "input_point_cloud_preview.png").exists()
    assert not (processed_dir / "debug_clusters.png").exists()
    skipped = [stage for stage in result["profiling"]["stages"] if stage["category"] == "debug_artifacts"]
    assert skipped
    assert all(stage["notes"] == "skipped by --skip-debug-images" for stage in skipped)
