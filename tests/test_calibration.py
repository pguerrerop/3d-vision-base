from __future__ import annotations

import json
from pathlib import Path

import open3d as o3d
import numpy as np
import pytest

from vision_3d_acquisition.calibration.calibration_models import (
    ObjectFilterConfig,
    PlaneCalibration,
    SystemCalibration,
)
from vision_3d_acquisition.calibration.calibration_storage import load_calibration, save_calibration
from vision_3d_acquisition.calibration.plane_detection import detect_plane_candidates, public_plane_candidate
from vision_3d_acquisition.processing.calibrated_filtering import (
    evaluate_clusters_with_calibration,
    filter_by_labeled_planes,
    points_inside_polygon_xy,
)
from vision_3d_acquisition.processing.pipeline import ProcessingConfig, process_take
from vision_3d_acquisition.utils.time import utc_now_iso


def make_cloud(points: list[tuple[float, float, float]]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def synthetic_planes() -> o3d.geometry.PointCloud:
    points: list[tuple[float, float, float]] = []
    for x in range(-15, 16):
        for y in range(-15, 16):
            points.append((float(x), float(y), 0.0))
    for x in range(-12, 13):
        for z in range(2, 18):
            points.append((float(x), 35.0, float(z)))
    return make_cloud(points)


def calibration(reference_take_id: str = "take_001") -> SystemCalibration:
    return SystemCalibration(
        calibration_id="belt_setup_test",
        reference_take_id=reference_take_id,
        planes=[
            PlaneCalibration(
                plane_id="plane_1",
                label="belt",
                plane_model=(0.0, 0.0, 1.0, 0.0),
                point_count=81,
                bbox_min_mm=(-20.0, -20.0, 0.0),
                bbox_max_mm=(20.0, 20.0, 0.0),
                extent_mm=(40.0, 40.0, 0.0),
                avg_z_mm=0.0,
                normal=(0.0, 0.0, 1.0),
                roi_polygon_xy_mm=[(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)],
            ),
            PlaneCalibration(
                plane_id="plane_2",
                label="outer_plane_ignore",
                plane_model=(0.0, 1.0, 0.0, -35.0),
                point_count=20,
                bbox_min_mm=(-12.0, 35.0, 2.0),
                bbox_max_mm=(12.0, 35.0, 17.0),
                extent_mm=(24.0, 0.0, 15.0),
                avg_z_mm=9.5,
                normal=(0.0, 1.0, 0.0),
                roi_polygon_xy_mm=[(-12.0, 35.0), (12.0, 35.0)],
            ),
        ],
        object_filter=ObjectFilterConfig(
            min_height_above_belt_mm=1.0,
            max_height_above_belt_mm=20.0,
            require_center_inside_belt=True,
            min_fraction_points_inside_belt=0.6,
        ),
    )


def write_take(data_dir: Path) -> tuple[str, Path]:
    take_id = "calibrated_take_001"
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True)
    points: list[tuple[float, float, float]] = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            points.append((float(x), float(y), 0.0))
    for base_x in (-12.0, 12.0):
        for dx in (0.0, 0.6, 1.2):
            for dy in (0.0, 0.6, 1.2):
                for dz in (0.0, 0.5):
                    points.append((base_x + dx, dy, 3.0 + dz))
    cloud = make_cloud(points)
    o3d.io.write_point_cloud(str(take_dir / "point_cloud.ply"), cloud)
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "created_at": utc_now_iso(),
                "files": {"point_cloud": "point_cloud.ply"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (take_dir / "READY").touch()
    return take_id, take_dir


def test_plane_candidate_detection_returns_ordered_candidates() -> None:
    candidates = detect_plane_candidates(
        synthetic_planes(),
        max_planes=2,
        min_plane_points=100,
        distance_threshold_mm=0.1,
    )

    public = [public_plane_candidate(candidate) for candidate in candidates]
    assert [candidate["plane_id"] for candidate in public] == ["plane_1", "plane_2"]
    assert public[0]["point_count"] > public[1]["point_count"]
    assert "plane_model" in public[0]
    assert len(public[0]["roi_polygon_xy_mm"]) >= 3
    assert "_cloud" not in public[0]


def test_calibration_save_load_round_trip(tmp_path: Path) -> None:
    original = calibration()

    path = save_calibration(original, tmp_path)
    loaded = load_calibration(path, tmp_path)

    assert path.name == "belt_setup_test.json"
    assert loaded.calibration_id == original.calibration_id
    assert loaded.belt_plane_3d().plane_id == "plane_1"
    assert loaded.belt_plane_3d().roi_polygon_xy_mm[0] == (-4.0, -4.0)


def test_only_one_belt_allowed() -> None:
    original = calibration()
    payload = original.model_dump()
    payload["planes"][1]["label"] = "belt"

    with pytest.raises(ValueError, match="exactly one belt"):
        SystemCalibration.model_validate(payload)


def test_calibration_json_validation() -> None:
    validated = SystemCalibration.model_validate_json(calibration().model_dump_json())

    assert validated.calibration_type == "plane_3d"
    assert validated.source_modalities == ["point_cloud"]
    assert validated.object_filter.min_fraction_points_inside_belt == 0.6
    assert validated.planes[0].plane_model == (0.0, 0.0, 1.0, 0.0)


def test_old_calibration_defaults_to_plane_3d() -> None:
    payload = calibration().model_dump(mode="json")
    payload.pop("calibration_type")
    payload.pop("source_modalities")

    validated = SystemCalibration.model_validate(payload)

    assert validated.calibration_type == "plane_3d"
    assert validated.source_modalities == ["point_cloud"]


def test_outer_planes_are_removed_and_belt_polygon_filters_points() -> None:
    cloud = make_cloud(
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 3.0),
            (12.0, 0.0, 3.0),
            (0.0, 35.0, 5.0),
        ]
    )

    filtered = filter_by_labeled_planes(cloud, calibration(), distance_threshold_mm=0.2)

    assert len(filtered["belt_reference_points"].points) == 1
    assert len(filtered["ignored_plane_points"].points) == 1
    assert len(filtered["candidate_foreground_points"].points) == 1
    assert filtered["statistics"]["ignored_plane_points"] == 1
    assert filtered["statistics"]["candidate_foreground_points"] == 1


def test_points_inside_polygon_xy() -> None:
    points = make_cloud([(0.0, 0.0, 1.0), (5.0, 0.0, 1.0), (-4.0, 0.0, 1.0)])

    mask = points_inside_polygon_xy(
        np.asarray(points.points),
        [(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)],
    )

    assert mask.tolist() == [True, False, True]


def test_clusters_outside_belt_are_rejected() -> None:
    outside_cluster = {"cluster_id": 1, "cloud": make_cloud([(8.0, 8.0, 3.0), (8.2, 8.0, 3.4), (8.0, 8.2, 3.2)])}

    kept, rejected = evaluate_clusters_with_calibration([outside_cluster], calibration(), min_cluster_points=1)

    assert kept == []
    assert len(rejected) == 1
    assert rejected[0]["calibration_metrics"]["filter_status"] == "rejected"
    assert rejected[0]["calibration_metrics"]["filter_reason"] == "center_outside_belt_polygon"


def test_processing_with_calibration_filters_objects_outside_belt(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = write_take(data_dir)
    output_dir = data_dir / "processed" / take_id

    result = process_take(
        take_dir,
        output_dir,
        ProcessingConfig(
            voxel_size=0.0,
            outlier_nb_neighbors=5,
            outlier_std_ratio=10.0,
            plane_distance_threshold=0.2,
            plane_num_iterations=200,
            dbscan_eps=1.6,
            dbscan_min_points=4,
            min_cluster_points=6,
        ),
        calibration=calibration(take_id),
    )

    assert result.calibration_id == "belt_setup_test"
    assert result.calibration is not None
    assert result.calibration.calibration_type == "plane_3d"
    assert result.algorithm_stage == "calibrated_segmentation"
    assert result.summary.object_count == 0
    assert result.plane_filtering is not None
    assert result.plane_filtering["belt_plane_id"] == "plane_1"
    assert result.plane_filtering["candidate_foreground_points"] == 0
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["calibration_id"] == "belt_setup_test"
    assert payload["calibration"]["calibration_type"] == "plane_3d"
    assert payload["input_modalities"] == ["point_cloud"]
    assert payload["algorithm_stage"] == "calibrated_segmentation"
    assert payload["plane_filtering"]["kept_objects"] == 0
    for filename in (
        "debug_calibrated_planes.png",
        "debug_belt_polygon_topview.png",
        "debug_filtered_foreground.png",
        "debug_rejected_points.png",
        "debug_clusters_filtered.png",
    ):
        assert (output_dir / filename).is_file()


def test_early_calibration_crop_reduces_points_before_preprocessing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = write_take(data_dir)
    output_dir = data_dir / "processed" / take_id

    result = process_take(
        take_dir,
        output_dir,
        ProcessingConfig(
            voxel_size=0.0,
            enable_outlier_removal=False,
            calibration_crop_margin_mm=1.0,
            plane_distance_threshold=0.2,
            plane_num_iterations=200,
            dbscan_eps=1.6,
            dbscan_min_points=4,
            min_cluster_points=6,
        ),
        calibration=calibration(take_id),
        skip_debug_images=True,
    )

    assert result.profiling is not None
    crop_stage = next(stage for stage in result.profiling.stages if stage.name == "early_calibration_crop")
    assert crop_stage.input_points is not None
    assert crop_stage.output_points is not None
    assert crop_stage.output_points < crop_stage.input_points
    voxel_stage = next(stage for stage in result.profiling.stages if stage.name == "voxel_downsample")
    assert voxel_stage.input_points == crop_stage.output_points
