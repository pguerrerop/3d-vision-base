from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext
from vision_3d_acquisition.vision_core.pipelines.stages_25d import (
    ClassifyMiningBall25DStage,
    _apply_known_object_scale_correction,
    _classify_25d,
    _contour_metrics_in_metric_space,
    _compose_diameter_metrics,
)
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


def test_classification_explanation_artifact_is_emitted(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}
    assert "classification_explanation" in by_id
    artifact = by_id["classification_explanation"]
    assert artifact.get("kind") == "json"
    assert artifact.get("stage_id") == "classification"
    assert artifact.get("path") == "classification_explanation.json"
    meta = artifact.get("metadata") or {}
    assert meta.get("semantic_type") == "classification_explanation"
    assert meta.get("artifact_type") == "classification_explanation"
    assert meta.get("scope") == "global"
    out = data_dir / "processed" / take_id / "classification_explanation.json"
    assert out.is_file()
    explanation = json.loads(out.read_text(encoding="utf-8"))
    assert explanation.get("artifact_id") == "classification_explanation"
    assert explanation.get("stage") == "classification"
    assert explanation.get("scope") == "global"
    assert isinstance(explanation.get("objects"), list)
    assert (payload.get("files") or {}).get("classification_explanation") == "classification_explanation.json"
    assert "metric_explanation" in by_id
    metric_art = by_id["metric_explanation"]
    assert metric_art.get("kind") == "json"
    assert metric_art.get("stage_id") == "classification"
    assert metric_art.get("path") == "metric_explanation.json"
    assert (payload.get("files") or {}).get("metric_explanation") == "metric_explanation.json"


def test_classification_explanation_flags_suspicious_diameter_inconsistency(tmp_path: Path) -> None:
    stage = ClassifyMiningBall25DStage()
    context = PipelineContext()
    context.set_artifact("height_processing_semantic_field", "height_above_belt")
    context.set_artifact("output_dir", tmp_path)
    context.set_artifact("files", {})
    context.set_artifact("objects", [{
        "object_id": 1,
        "dimensions_mm": [83.6, 82.0, 74.92],
        "major_axis_mm": 83.6,
        "minor_axis_mm": 82.0,
        "diameter_mm": 385.35,
        "sphericity_score": 0.001,
        "feature_sphericity_3d": 0.0444,
        "feature_eccentricity": 0.999,
        "feature_flatness": -0.2,
        "feature_edge_roughness": 10.0,
        "feature_local_curvature_proxy": 47.4,
        "feature_volume_proxy_mm3": 332581.0,
        "height_above_belt_mm": {"max_height_mm": 74.9, "mean_height_mm": 63.6, "p95_height_mm": 74.0, "height_std_mm": 1.0},
    }])
    stage.run(context)
    obj = context.get_artifact("objects", [])[0]
    explanation = (obj.get("classification_explanation") or {})
    rules = explanation.get("rules") or []
    mismatch = next((item for item in rules if item.get("rule_id") == "sanity.diameter_vs_dims"), None)
    assert mismatch is not None
    assert mismatch.get("passed") is False
    assert mismatch.get("severity") in ("warning", "critical")
    assert "Suspicious diameter mismatch" in str(mismatch.get("message") or "")


def test_classification_explanation_has_decisive_rule_for_each_object(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    objects = payload.get("objects") or []
    assert objects
    # Per-object classification_explanation is stripped from the DetectedObject
    # contract, so consult the global classification_explanation artifact which
    # carries the full rule trace for every object.
    explanation_path = data_dir / "processed" / take_id / "classification_explanation.json"
    assert explanation_path.is_file()
    explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
    explained_objects = explanation.get("objects") or []
    assert len(explained_objects) == len(objects)
    for exp in explained_objects:
        summary = (exp.get("decision_summary") or {})
        decisive = summary.get("decisive_rule_ids") or []
        assert isinstance(decisive, list)
        assert len(decisive) >= 1


def test_known_cube_correction_not_double_applied() -> None:
    objects = [{
        "object_id": 1,
        "dimensions_mm": [10.0, 20.0, 30.0],
        "major_axis_mm": 20.0,
        "minor_axis_mm": 10.0,
        "diameter_mm": 20.0,
        "height_above_belt_mm": {"max_height_mm": 30.0, "mean_height_mm": 15.0, "p95_height_mm": 29.0, "height_std_mm": 1.0},
        "feature_volume_proxy_mm3": 6000.0,
        "feature_edge_roughness": 1.0,
        "footprint_area_mm2": 200.0,
    }]
    changed_first = _apply_known_object_scale_correction(objects, scale_x=2.0, scale_y=2.0, scale_z=2.0, correction_source="test", correction_context_id="ctx")
    dims_once = tuple(objects[0]["dimensions_mm"])
    changed_second = _apply_known_object_scale_correction(objects, scale_x=2.0, scale_y=2.0, scale_z=2.0, correction_source="test", correction_context_id="ctx")
    dims_twice = tuple(objects[0]["dimensions_mm"])
    assert changed_first is True
    assert changed_second is False
    assert dims_once == dims_twice
    assert objects[0].get("metrics_coordinate_space") == "corrected_mm"


def test_diameter_sanity_detects_impossible_selected_value_and_falls_back() -> None:
    item = {
        "dimensions_mm": [83.6, 82.0, 74.92],
        "major_axis_mm": 83.6,
        "minor_axis_mm": 82.0,
        "diameter_mm": 385.35,
        "footprint_area_mm2": 5400.0,
    }
    d = _compose_diameter_metrics(item)
    assert d["diameter_sanity_status"] in ("suspicious", "invalid")
    assert "inconsistent" in str(d["diameter_sanity_message"]).lower() or "impossible" in str(d["diameter_sanity_message"]).lower()
    assert float(d["diameter_selected_mm"]) < 200.0
    assert str(d["diameter_selected_source"]).endswith("fallback_from_sanity")


def test_classification_explanation_contains_raw_and_corrected_values(tmp_path: Path) -> None:
    stage = ClassifyMiningBall25DStage()
    context = PipelineContext()
    context.set_artifact("height_processing_semantic_field", "height_above_belt")
    context.set_artifact("output_dir", tmp_path)
    context.set_artifact("files", {})
    context.set_artifact("objects", [{
        "object_id": 1,
        "class_name": "non_ball",
        "label": "chatarra",
        "superclass": "SCRAP",
        "confidence": 0.78,
        "dimensions_mm": [83.6, 82.0, 74.92],
        "major_axis_mm": 83.6,
        "minor_axis_mm": 82.0,
        "diameter_mm": 120.0,
        "sphericity_score": 0.10,
        "feature_sphericity_3d": 0.20,
        "feature_eccentricity": 0.7,
        "feature_flatness": 0.1,
        "feature_edge_roughness": 10.0,
        "feature_local_curvature_proxy": 20.0,
        "feature_volume_proxy_mm3": 10000.0,
        "height_above_belt_mm": {"max_height_mm": 70.0, "mean_height_mm": 60.0, "p95_height_mm": 69.0, "height_std_mm": 2.0},
        "scale_correction_applied": {"x": 1.1, "y": 1.2, "z": 1.3},
        "metrics_coordinate_space": "corrected_mm",
        "correction_source": "persisted_known_cube",
        "raw_metrics": {
            "major_axis_mm": 76.0,
            "minor_axis_mm": 68.0,
            "diameter_mm": 90.0,
            "sphericity_score": 0.2,
            "feature_sphericity_3d": 0.3,
            "feature_eccentricity": 0.6,
            "feature_flatness": 0.2,
            "feature_edge_roughness": 7.0,
            "feature_volume_proxy_mm3": 8000.0,
            "height_above_belt_mm": {"max_height_mm": 55.0, "mean_height_mm": 45.0, "p95_height_mm": 54.0},
        },
    }])
    stage.run(context)
    obj = context.get_artifact("objects", [])[0]
    explanation = obj.get("classification_explanation") or {}
    rules = explanation.get("rules") or []
    assert rules
    assert any(("raw_value" in row and "corrected_value" in row and "correction_scale" in row) for row in rules if isinstance(row, dict))
    traces = explanation.get("metric_trace") or []
    assert traces
    required = {
        "metric_key",
        "final_value",
        "raw_value",
        "corrected_value",
        "correction_applied",
        "correction_scales",
        "coordinate_space_before",
        "coordinate_space_after",
        "source_artifact_id",
        "source_stage",
        "formula_name",
        "formula_human_readable",
        "formula_inputs",
        "intermediate_values",
        "validity_status",
        "warnings",
        "used_by_classifier",
        "correction_factor_used",
        "geometry_metric_source",
    }
    assert required.issubset(set(traces[0].keys()))
    by_key = {str(row.get("metric_key") or ""): row for row in traces}
    # New per-metric traces must be present so the Studio Metric details tab
    # can show the corrected acquisition->mm transformation chain.
    for expected in (
        "feature_eccentricity",
        "sphericity_score",
        "feature_sphericity_3d",
        "diameter_selected_mm",
        "feature_local_curvature_proxy",
        "max_height_mm",
        "footprint_area_mm2",
        "feature_volume_proxy_mm3",
        "scale_correction_applied",
    ):
        assert expected in by_key, f"metric_trace missing key {expected}"
    # Correction factor must reflect the appropriate per-metric scale.
    assert by_key["max_height_mm"]["correction_factor_used"] == 1.3
    assert by_key["feature_volume_proxy_mm3"]["correction_factor_used"] == pytest.approx(1.1 * 1.2 * 1.3)
    assert by_key["footprint_area_mm2"]["correction_factor_used"] == pytest.approx(1.1 * 1.2)
    # Calibration context entry surfaces the applied correction source so
    # operators can verify which calibration produced the corrected metrics.
    calibration = by_key["scale_correction_applied"]
    calibration_inputs = {entry.get("name"): entry.get("value") for entry in calibration["formula_inputs"] if isinstance(entry, dict)}
    assert calibration_inputs.get("correction_source") == "persisted_known_cube"
    assert calibration["correction_applied"] is True
    # feature_local_curvature_proxy now surfaces the per-axis correction inputs.
    curvature_inputs = {entry.get("name"): entry.get("value") for entry in by_key["feature_local_curvature_proxy"]["formula_inputs"] if isinstance(entry, dict)}
    assert {"scale_x", "scale_y", "scale_z"}.issubset(curvature_inputs.keys())


def test_classification_explanation_written_for_single_non_ball_object(tmp_path: Path) -> None:
    stage = ClassifyMiningBall25DStage()
    context = PipelineContext()
    context.set_artifact("height_processing_semantic_field", "height_above_belt")
    context.set_artifact("output_dir", tmp_path)
    context.set_artifact("files", {})
    context.set_artifact("objects", [{
        "object_id": 1,
        "class_name": "unknown",
        "label": "unknown",
        "superclass": "SCRAP",
        "confidence": 0.7,
        "dimensions_mm": [83.6, 82.0, 74.92],
        "major_axis_mm": 83.6,
        "minor_axis_mm": 82.0,
        "diameter_mm": 385.35,
        "sphericity_score": 0.001,
        "feature_sphericity_3d": 0.0444,
        "feature_eccentricity": 0.999,
        "feature_flatness": -0.2,
        "feature_edge_roughness": 10.0,
        "feature_local_curvature_proxy": 47.4,
        "feature_volume_proxy_mm3": 332581.0,
        "height_above_belt_mm": {"max_height_mm": 74.9, "mean_height_mm": 63.6, "p95_height_mm": 74.0, "height_std_mm": 1.0},
    }])
    stage.run(context)
    out = tmp_path / "classification_explanation.json"
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    objs = payload.get("objects") or []
    assert len(objs) == 1
    assert objs[0].get("final_class_name") == "non_ball"


def test_near_round_contour_metrics_are_physically_coherent() -> None:
    # Circle-like contour in pixel space; anisotropy-aware mm conversion should
    # keep eccentricity low after correction.
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    contour = [[float(100 + 20 * np.cos(t)), float(120 + 19 * np.sin(t))] for t in theta]
    m = _contour_metrics_in_metric_space(contour, x_resolution_mm=1.0, y_resolution_mm=1.0, scale_x=1.0, scale_y=1.0)
    assert m.get("valid") is True
    assert float(m.get("eccentricity")) < 0.35
    assert float(m.get("roundness")) > 0.75


def test_corrected_geometry_differs_from_raw_with_anisotropic_scaling() -> None:
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    contour = [[float(80 + 18 * np.cos(t)), float(80 + 18 * np.sin(t))] for t in theta]
    raw = _contour_metrics_in_metric_space(contour, x_resolution_mm=1.0, y_resolution_mm=0.2, scale_x=1.0, scale_y=1.0)
    corrected = _contour_metrics_in_metric_space(contour, x_resolution_mm=1.0, y_resolution_mm=0.2, scale_x=1.0, scale_y=5.0)
    assert raw.get("valid") is True and corrected.get("valid") is True
    assert abs(float(raw.get("eccentricity")) - float(corrected.get("eccentricity"))) > 0.05


def test_sphericity_3d_is_recomputed_from_corrected_dimensions_after_correction() -> None:
    # Mirrors the real-world TriSpector dataset where raw bbox is highly
    # anisotropic (88x410x2052) and the persisted known-cube correction
    # is (scale_x, scale_y, scale_z) ~= (0.95, 0.2, 0.0365), so corrected
    # extents land at (~83.6, ~82.0, ~74.9). feature_sphericity_3d MUST
    # follow the corrected XYZ extents (~0.9), NOT the stretched ellipse
    # axes that collapsed it to ~0.044 in the bug report.
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    contour = [[float(100 + 44 * np.cos(t)), float(220 + 205 * np.sin(t))] for t in theta]
    objects = [{
        "object_id": 1,
        "dimensions_mm": [88.0, 410.0, 2052.0],
        "major_axis_mm": 410.0,
        "minor_axis_mm": 88.0,
        "diameter_mm": 410.0,
        "feature_eccentricity": 0.978,
        "sphericity_score": 0.214,
        "feature_volume_proxy_mm3": 1e9,
        "feature_edge_roughness": 50.0,
        "feature_local_curvature_proxy": 1300.0,
        "feature_curvature_components_raw": {
            "mean_abs_gx_mm_per_mm": 30.0,
            "mean_abs_gy_mm_per_mm": 30.0,
            "x_resolution_mm": 1.0,
            "y_resolution_mm": 1.0,
            "computed_over": "inner_mask",
        },
        "footprint_area_mm2": 36000.0,
        "contour_px": contour,
        "geometry_metric_context": {"x_resolution_mm": 1.0, "y_resolution_mm": 1.0},
        "height_above_belt_mm": {"max_height_mm": 2052.0, "mean_height_mm": 1742.0, "p95_height_mm": 2025.0, "height_std_mm": 220.0},
    }]
    _apply_known_object_scale_correction(
        objects,
        scale_x=0.95,
        scale_y=0.2,
        scale_z=0.0365,
        correction_source="test",
        correction_context_id="ctx",
    )
    row = objects[0]
    # Corrected dimensions are physically coherent.
    dims = row["dimensions_mm"]
    assert abs(float(dims[0]) - 83.6) < 0.1
    assert abs(float(dims[1]) - 82.0) < 0.5
    assert abs(float(dims[2]) - 74.9) < 0.5
    # Corrected 3D axis balance reflects min/max of corrected XYZ extents.
    sph3d = float(row["feature_sphericity_3d"])
    assert sph3d > 0.7, f"feature_sphericity_3d should be > 0.7 for near-round XYZ extents, got {sph3d}"
    # Corrected eccentricity (from contour) is low for the round contour.
    assert float(row["feature_eccentricity"]) < 0.45
    # Corrected curvature uses scale_z/scale_x and scale_z/scale_y, not just scale_z.
    corrected_curv = float(row["feature_local_curvature_proxy"])
    expected = 30.0 * (0.0365 / 0.95) + 30.0 * (0.0365 / 0.2)
    assert abs(corrected_curv - expected) < 1e-2, f"expected {expected:.4f}, got {corrected_curv:.4f}"
    # Anisotropic warning should NOT fire when corrected geometry is valid.
    warnings = row.get("geometry_invariant_warnings") or []
    assert not any("anisotropic" in str(w) for w in warnings), (
        f"anisotropic warning should be suppressed when contour-based corrected geometry is valid; got {warnings}"
    )


def test_near_round_object_classifies_as_ball_after_correction() -> None:
    # End-to-end: a near-round object produced from a raw heightmap with strongly
    # anisotropic calibration should ultimately classify as a ball once corrected.
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    contour = [[float(100 + 44 * np.cos(t)), float(220 + 205 * np.sin(t))] for t in theta]
    objects = [{
        "object_id": 1,
        "dimensions_mm": [88.0, 410.0, 2052.0],
        "major_axis_mm": 410.0,
        "minor_axis_mm": 88.0,
        "diameter_mm": 410.0,
        "feature_eccentricity": 0.978,
        "sphericity_score": 0.214,
        "feature_flatness": 0.87,
        "feature_edge_roughness": 3.66,
        "feature_volume_proxy_mm3": 5.0e10,
        "feature_local_curvature_proxy": 1300.0,
        "feature_curvature_components_raw": {
            "mean_abs_gx_mm_per_mm": 30.0,
            "mean_abs_gy_mm_per_mm": 30.0,
            "x_resolution_mm": 1.0,
            "y_resolution_mm": 1.0,
            "computed_over": "inner_mask",
        },
        "footprint_area_mm2": 36000.0,
        "contour_px": contour,
        "geometry_metric_context": {"x_resolution_mm": 1.0, "y_resolution_mm": 1.0},
        "height_above_belt_mm": {"max_height_mm": 2052.0, "mean_height_mm": 1742.0, "p95_height_mm": 2025.0, "height_std_mm": 220.0},
    }]
    _apply_known_object_scale_correction(
        objects, scale_x=0.95, scale_y=0.2, scale_z=0.0365,
        correction_source="test", correction_context_id="ctx",
    )
    row = objects[0]
    label, display_label, class_name, confidence = _classify_25d(row)
    assert class_name == "ball", (
        f"near-round corrected object should classify as ball, got {class_name} "
        f"(sph3d={row.get('feature_sphericity_3d')}, ecc={row.get('feature_eccentricity')})"
    )


def test_anisotropic_warning_only_when_contour_correction_unavailable() -> None:
    # Same aggressive anisotropic scaling, but without a contour_px the
    # contour-based corrected geometry cannot be derived: the explanation
    # must surface the anisotropic-scale advisory.
    objects = [{
        "object_id": 1,
        "dimensions_mm": [88.0, 410.0, 2052.0],
        "major_axis_mm": 410.0,
        "minor_axis_mm": 88.0,
        "diameter_mm": 410.0,
        "feature_eccentricity": 0.5,
        "sphericity_score": 0.5,
        "feature_volume_proxy_mm3": 1e9,
        "feature_edge_roughness": 50.0,
        "footprint_area_mm2": 36000.0,
        "height_above_belt_mm": {"max_height_mm": 2052.0, "mean_height_mm": 1742.0, "p95_height_mm": 2025.0, "height_std_mm": 220.0},
    }]
    _apply_known_object_scale_correction(
        objects, scale_x=0.95, scale_y=0.2, scale_z=0.0365,
        correction_source="test", correction_context_id="ctx",
    )
    warnings = objects[0].get("geometry_invariant_warnings") or []
    assert any("anisotropic" in str(w) for w in warnings), (
        f"anisotropic warning should fire when contour-based geometry is unavailable; got {warnings}"
    )


def test_geometry_debug_artifacts_are_emitted_with_correction(tmp_path: Path) -> None:
    objects = [{
        "object_id": 1,
        "dimensions_mm": [10.0, 10.0, 10.0],
        "major_axis_mm": 10.0,
        "minor_axis_mm": 10.0,
        "diameter_mm": 10.0,
        "height_above_belt_mm": {"max_height_mm": 10.0, "mean_height_mm": 8.0, "p95_height_mm": 9.0, "height_std_mm": 1.0},
        "feature_volume_proxy_mm3": 1000.0,
        "feature_edge_roughness": 1.0,
        "footprint_area_mm2": 80.0,
        "contour_px": [[10.0, 10.0], [11.0, 10.0], [12.0, 11.0], [11.0, 12.0], [10.0, 12.0], [9.0, 11.0]],
        "geometry_metric_context": {"x_resolution_mm": 1.0, "y_resolution_mm": 0.2},
    }]
    _apply_known_object_scale_correction(objects, scale_x=1.0, scale_y=5.0, scale_z=1.0, correction_source="test", correction_context_id="ctx")
    assert objects[0].get("geometry_coordinate_space") == "corrected_metric_mm"
    assert isinstance(objects[0].get("geometry_debug"), dict)
    assert objects[0].get("geometry_debug", {}).get("corrected", {}).get("valid") is True


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
