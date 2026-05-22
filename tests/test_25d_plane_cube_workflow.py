from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vision_3d_acquisition.api.main import latest_operator_inspection_result, publish_25d_take_result
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.apps.ball_inspection_25d.pipeline import run_ball_inspection_25d_flow
from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, save_heightmap_npz


def _settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def _write_take(settings: ApiSettings, take_id: str, frame: HeightmapFrame, *, known_object: dict | None = None) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    save_heightmap_npz(frame, take_dir / "heightmap.npz")
    meta = {
        "take_id": take_id,
        "source": "trispector_ftp_0",
        "created_at": "2026-05-22T10:00:00Z",
        "modalities": ["heightmap"],
        "files": {"heightmap": "heightmap.npz"},
    }
    if known_object:
        meta["known_object_25d"] = known_object
    (take_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (take_dir / "READY").touch()


def _tilted_plane_with_cube(width: int = 160, height: int = 120) -> HeightmapFrame:
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    plane = 100.0 + 0.03 * xx + 0.02 * yy
    cube_mask = (xx > 50) & (xx < 90) & (yy > 40) & (yy < 80)
    z = plane.copy()
    z[cube_mask] += 25.0
    valid = np.ones_like(z, dtype=bool)
    z[10:20, 10:20] = 0.0
    valid[10:20, 10:20] = False
    return HeightmapFrame(
        z_mm=z.astype(np.float32),
        valid_mask=valid,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="sensor_xy_z_mm",
    )


def test_plane_normalization_and_cube_scale_validation(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    frame = _tilted_plane_with_cube()
    _write_take(
        settings,
        "take_cube",
        frame,
        known_object={
            "enabled": True,
            "object_label": "cubo",
            "known_width_mm": 40.0,
            "known_depth_mm": 40.0,
            "known_height_mm": 25.0,
            "tolerance_percent": 35.0,
            "target_selection": "largest_component",
        },
    )
    result = run_ball_inspection_25d_flow(settings.data_dir, take_id="take_cube")
    out = result.output_dir
    plane_debug = json.loads((out / "plane_fit_debug.json").read_text(encoding="utf-8"))
    norm_debug = json.loads((out / "normalization_debug.json").read_text(encoding="utf-8"))
    scale = json.loads((out / "known_object_scale_validation.json").read_text(encoding="utf-8"))
    assert plane_debug["status"] in {"success", "failed"}
    assert norm_debug["background_height_p95_abs_after_normalization_mm"] < 2.5
    assert abs(float(scale["measured_height_mm"]) - 25.0) < 6.0


def test_failed_plane_fit_still_emits_debug(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    z = np.zeros((80, 80), dtype=np.float32)
    valid = np.zeros_like(z, dtype=bool)
    valid[0:4, 0:4] = True
    z[0:4, 0:4] = 12.0
    frame = HeightmapFrame(
        z_mm=z,
        valid_mask=valid,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="sensor_xy_z_mm",
    )
    _write_take(settings, "take_fail_plane", frame)
    result = run_ball_inspection_25d_flow(settings.data_dir, take_id="take_fail_plane")
    out = result.output_dir
    plane_debug = json.loads((out / "plane_fit_debug.json").read_text(encoding="utf-8"))
    assert plane_debug["status"] == "failed"
    assert (out / "plane_inlier_mask.png").is_file()
    assert (out / "normalized_heightmap.png").is_file()


def test_stage_artifacts_include_plane_debug_ids(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _write_take(settings, "take_artifacts", _tilted_plane_with_cube())
    result = run_ball_inspection_25d_flow(settings.data_dir, take_id="take_artifacts")
    artifacts = result.result_payload.get("artifacts") or []
    ids = {str(item.get("artifact_id")) for item in artifacts if isinstance(item, dict)}
    assert "plane_fit_debug" in ids
    assert "normalized_height_histogram" in ids
    assert "plane_inlier_mask" in ids
    assert "background_candidate_mask" in ids
    assert "background_selection_debug" in ids


def test_background_candidate_selection_prefers_border_plane(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    width, height = 180, 140
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    plane = 100.0 + 0.05 * xx + 0.03 * yy
    z = plane.copy()
    # Central protruding object should not become background candidate.
    obj_mask = (xx > 60) & (xx < 120) & (yy > 45) & (yy < 105)
    z[obj_mask] += 28.0
    valid = np.ones_like(z, dtype=bool)
    frame = HeightmapFrame(
        z_mm=z.astype(np.float32),
        valid_mask=valid,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="sensor_xy_z_mm",
    )
    _write_take(settings, "take_bg_candidates", frame)
    result = run_ball_inspection_25d_flow(settings.data_dir, take_id="take_bg_candidates")
    out = result.output_dir
    plane_debug = json.loads((out / "plane_fit_debug.json").read_text(encoding="utf-8"))
    bg_debug = json.loads((out / "background_selection_debug.json").read_text(encoding="utf-8"))
    norm_debug = json.loads((out / "normalization_debug.json").read_text(encoding="utf-8"))
    seg_debug = json.loads((out / "segmentation_debug.json").read_text(encoding="utf-8"))
    assert (out / "background_candidate_mask.png").is_file()
    assert (out / "background_depth_histogram.json").is_file()
    assert bg_debug["selected_candidate_pixel_count"] > 100
    assert bg_debug["candidate_coverage_percent"] > 2.0
    assert plane_debug["inlier_ratio"] > 0.2
    assert norm_debug["background_height_p95_abs_after_normalization_mm"] < 3.5
    assert norm_debug["foreground_mean_height_mm"] > 4.0
    assert seg_debug["foreground_coverage_percent"] > 0.0


def test_publish_25d_result_and_operator_latest(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _write_take(settings, "take_publish", _tilted_plane_with_cube())
    run_ball_inspection_25d_flow(settings.data_dir, take_id="take_publish")
    published = publish_25d_take_result("take_publish", settings=settings)
    assert published["published_result_id"]
    assert published["source_refs"]["result_mode"] == "25d_only"
    latest = latest_operator_inspection_result(settings=settings)
    assert latest["published_result_id"] == published["published_result_id"]
