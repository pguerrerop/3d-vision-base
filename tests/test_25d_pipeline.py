from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext
from vision_3d_acquisition.vision_core.pipelines.stages_25d import ClassifyMiningBall25DStage
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
        assert isinstance(obj.get("superclass"), str)
        assert isinstance(obj.get("label"), str)
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
    classification = payload.get("classification") or {}
    assert isinstance(summary.get("object_count"), int)
    assert isinstance(summary.get("superclass"), str)
    assert isinstance(summary.get("label"), str)
    assert isinstance(classification.get("superclass"), str)
    assert isinstance(classification.get("label"), str)
    assert payload.get("processing_pipeline", {}).get("pipeline_family") == "25d"
    assert isinstance(payload.get("object_candidates"), list)


def test_reference_surface_roi_does_not_crop_segmentation_or_normalization(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    roi_stage_params = {
        "detect_belt_plane": {
            "plane_fit_roi": {"enabled": True, "x": 120, "y": 120, "width": 120, "height": 120},
        },
    }
    run_ball_inspection_25d_flow(data_dir, take_id=take_id, stage_params=roi_stage_params)
    payload = _load_result(data_dir, take_id)
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}
    plane_fit_debug = (by_id.get("plane_fit_debug", {}).get("metadata") or {}) if isinstance(by_id.get("plane_fit_debug"), dict) else {}
    assert plane_fit_debug.get("roi_enabled") is True
    assert plane_fit_debug.get("roi_x") == 120
    assert plane_fit_debug.get("roi_y") == 120
    assert plane_fit_debug.get("roi_width") == 120
    assert plane_fit_debug.get("roi_height") == 120
    assert isinstance(plane_fit_debug.get("roi_area_ratio"), float)
    seg_debug = (by_id.get("segmentation_debug", {}).get("metadata") or {}) if isinstance(by_id.get("segmentation_debug"), dict) else {}
    assert "roi" not in seg_debug


def test_reference_surface_vertical_band_roi_uses_full_height(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    frame = load_heightmap_npz(take_dir / "heightmap.npz")
    roi_stage_params = {
        "detect_belt_plane": {
            "plane_fit_roi": {"enabled": True, "type": "vertical_band", "x": 80, "width": 90},
        },
    }
    run_ball_inspection_25d_flow(data_dir, take_id=take_id, stage_params=roi_stage_params)
    payload = _load_result(data_dir, take_id)
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}
    plane_fit_debug = (by_id.get("plane_fit_debug", {}).get("metadata") or {}) if isinstance(by_id.get("plane_fit_debug"), dict) else {}
    assert plane_fit_debug.get("roi_enabled") is True
    assert plane_fit_debug.get("roi_type") == "vertical_band"
    assert plane_fit_debug.get("roi_x") == 80
    assert plane_fit_debug.get("roi_y") == 0
    assert plane_fit_debug.get("roi_width") == 90
    assert plane_fit_debug.get("roi_height") == int(frame.z_mm.shape[0])


def test_processing_artifacts_mark_display_only_vs_numeric_source(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    payload = _load_result(data_dir, take_id)
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}

    norm = by_id["normalized_heightmap"]
    norm_display = by_id["normalized_heightmap_display"]
    norm_render_context = by_id["normalized_heightmap_render_context"]
    seg = by_id["height_segmentation"]
    cls = by_id["classification_result_25d"]

    assert norm.get("metadata", {}).get("display_only") is True
    assert norm.get("metadata", {}).get("numeric_source") is False
    assert norm_display.get("metadata", {}).get("display_only") is False
    assert norm_display.get("metadata", {}).get("numeric_source") is True
    assert norm_render_context.get("metadata", {}).get("render_context_for") == "normalized_heightmap.png"
    assert norm.get("metadata", {}).get("semantic_field") == "height_above_belt"
    assert norm_display.get("metadata", {}).get("semantic_field") == "preview_normalized"
    assert seg.get("metadata", {}).get("numeric_source_artifact") in {"normalized_heightmap.npz", "height_above_belt_raster"}
    assert cls.get("metadata", {}).get("numeric_source_artifact") in {"normalized_heightmap.npz", "height_above_belt_raster"}
    assert norm.get("metadata", {}).get("derived_from") == "plane_signed_distance"
    assert (norm.get("metadata", {}).get("transform") or {}).get("type") == "invert_signed_distance"


def test_measurement_and_classification_contract_use_height_above_belt(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    assert payload.get("status") == "ok"
    artifacts = payload.get("artifacts") or []
    measurement_items = [
        item
        for item in artifacts
        if isinstance(item, dict) and str(item.get("artifact_id", "")).startswith("measurement_object_")
    ]
    classification_items = [
        item
        for item in artifacts
        if isinstance(item, dict) and str(item.get("artifact_id", "")).startswith("classification_object_")
    ]
    assert measurement_items
    assert classification_items
    for item in measurement_items + classification_items:
        assert item.get("metadata", {}).get("semantic_field") == "height_above_belt"


def test_color_mapping_block_is_persisted_canonically(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    payload = _load_result(data_dir, take_id)
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}

    # Both the render context, the display metadata, the rendered preview and
    # the canonical raster artifact must carry the same color_mapping contract
    # so the renderer, HeightLegend and hover diffing share one source of truth.
    norm = by_id["normalized_heightmap"]
    norm_display = by_id["normalized_heightmap_display"]
    norm_render_ctx = by_id["normalized_heightmap_render_context"]
    raster = by_id["height_above_belt_raster"]
    for name, item in (
        ("normalized_heightmap", norm),
        ("normalized_heightmap_display", norm_display),
        ("normalized_heightmap_render_context", norm_render_ctx),
        ("height_above_belt_raster", raster),
    ):
        meta = item.get("metadata") or {}
        block = meta.get("color_mapping") or {}
        assert block, f"missing color_mapping on {name}"
        for key in (
            "semantic_field",
            "value_min",
            "value_max",
            "color_scale_min",
            "color_scale_max",
            "color_map",
            "direction",
            "clamp",
            "source",
        ):
            assert key in block, f"{name} color_mapping missing {key}"
        assert block["semantic_field"] == "height_above_belt"
        assert block["direction"] in ("higher_is_hotter", "lower_is_hotter")
        assert block["color_map"] == "turbo"

    # The render context's color_mapping should match the renderer source of
    # truth and be flagged as source=render_context.
    ctx_block = (norm_render_ctx.get("metadata") or {}).get("color_mapping") or {}
    assert ctx_block.get("source") == "render_context"
    assert float(ctx_block.get("color_scale_min")) == float((norm_render_ctx.get("metadata") or {}).get("render_vmin"))
    assert float(ctx_block.get("color_scale_max")) == float((norm_render_ctx.get("metadata") or {}).get("render_vmax"))


def test_canonical_semantic_rasters_are_persisted(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    out = data_dir / "processed" / take_id
    assert (out / "raw_sensor_z.values.f32").is_file()
    assert (out / "plane_signed_distance.values.f32").is_file()
    assert (out / "height_above_belt.values.f32").is_file()


def test_classification_rejects_preview_semantic_input() -> None:
    stage = ClassifyMiningBall25DStage()
    context = PipelineContext()
    context.set_artifact("height_processing_semantic_field", "preview_normalized")
    context.set_artifact("objects", [])
    try:
        stage.run(context)
    except ValueError as exc:
        assert "cannot consume preview_normalized" in str(exc)
    else:
        raise AssertionError("Expected ValueError for preview_normalized classification input")


def test_preview_png_changes_do_not_affect_numeric_processing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    preview_path = take_dir / "heightmap_preview.png"
    preview_path.write_bytes(b"\x89PNG\r\n\x1a\nNOT_USED_BY_NUMERIC_PIPELINE")

    first = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    first_objects = first.get("objects") or []
    assert first_objects
    first_h = float((first_objects[0].get("height_above_belt_mm") or {}).get("max_height_mm") or 0.0)

    # Change preview bytes again; numeric result should remain stable.
    preview_path.write_bytes(b"\x89PNG\r\n\x1a\nDIFFERENT_PREVIEW_BYTES")
    second = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    second_objects = second.get("objects") or []
    assert second_objects
    second_h = float((second_objects[0].get("height_above_belt_mm") or {}).get("max_height_mm") or 0.0)
    assert abs(first_h - second_h) < 1e-6


def test_corrupt_or_missing_preview_does_not_break_processing_when_npz_exists(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    metadata_path = take_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.get("files") or {}
    files["heightmap_preview"] = "heightmap_preview.png"
    metadata["files"] = files
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Do not create the preview on purpose.
    result = run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    assert result.result_payload.get("status") == "ok"


def test_numeric_height_source_rejects_preview_png(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, take_dir = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    bad_source = take_dir / "heightmap_bad.png"
    bad_source.write_bytes(b"\x89PNG\r\n\x1a\nNOT_A_NUMERIC_SOURCE")
    metadata_path = take_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["heightmap"] = "heightmap_bad.png"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    except ValueError as exc:
        assert "Preview image cannot be used as numeric height source" in str(exc)
    else:
        raise AssertionError("Expected ValueError when PNG is used as numeric height source")
