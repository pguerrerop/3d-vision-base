from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from vision_3d_acquisition.vision_core.heightmap import load_heightmap_npz


def _load_result(data_dir: Path, take_id: str) -> dict:
    path = data_dir / "processed" / take_id / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_create_synthetic_25d_take_and_load(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    assert (take_dir / "READY").is_file()
    frame = load_heightmap_npz(take_dir / "heightmap.npz")
    assert frame.z_mm.shape[0] > 0 and frame.z_mm.shape[1] > 0
    assert np.count_nonzero(frame.valid_mask) > 0
    assert np.count_nonzero(~frame.valid_mask) > 0
    assert take_id


def test_25d_pipeline_plane_normalization_segmentation_measurement(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    result = run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    payload = _load_result(data_dir, take_id)

    assert result.result_payload.get("processing_pipeline", {}).get("id") == "mining_steel_ball_classification_25d"
    assert payload.get("status") == "ok"

    plane = payload.get("plane_model")
    assert isinstance(plane, list) and len(plane) == 4
    a, b, c, _d = [float(v) for v in plane]
    assert abs(c) > 0.5
    slope_x = -a / c
    slope_y = -b / c
    assert abs(slope_x) > 0.001
    assert abs(slope_y) > 0.001

    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}
    assert "normalized_heightmap" in by_id
    assert "belt_plane" in by_id
    assert "height_segmentation" in by_id
    assert "height_segmentation_overlay" in by_id
    assert "measurement_overlay" in by_id
    assert "classification_overlay" in by_id
    assert by_id["height_segmentation_overlay"].get("overlay_type") == "segmentation"
    assert by_id["height_segmentation_overlay"].get("coordinate_space") == "image_pixel"
    assert by_id["height_segmentation_overlay"].get("target_artifact_id") == "heightmap_preview"

    objects = payload.get("objects") or []
    assert len(objects) >= 2
    for obj in objects:
        heights = obj.get("height_above_belt_mm") or {}
        assert "max_height_mm" in heights
        assert "mean_height_mm" in heights
        assert "p95_height_mm" in heights
    object_candidates = payload.get("object_candidates")
    assert isinstance(object_candidates, list)
    if object_candidates:
        assert object_candidates[0].get("source_modality") == "derived_25d"


def test_25d_result_payload_contains_expected_fields(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    payload = _load_result(data_dir, take_id)

    files = payload.get("files") or {}
    assert files.get("heightmap")
    assert files.get("heightmap_npz")
    assert files.get("normalized_heightmap")
    assert files.get("overlay")

    summary = payload.get("summary") or {}
    assert isinstance(summary.get("object_count"), int)
    assert payload.get("processing_pipeline", {}).get("pipeline_family") == "25d"
    assert isinstance(payload.get("object_candidates"), list)
