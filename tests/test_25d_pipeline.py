from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext
from vision_3d_acquisition.vision_core.pipelines.stages_25d import (
    ClassifyMiningBall25DStage,
    _apply_known_object_scale_correction,
    _classify_25d,
    _compose_diameter_metrics,
    _contour_metrics_in_metric_space,
    _summarize_low_gradient_blobs,
)
from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, load_heightmap_npz, save_heightmap_npz


def _load_result(data_dir: Path, take_id: str) -> dict:
    path = data_dir / "processed" / take_id / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mask_png(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None, path
    return mask > 0


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
        assert isinstance(obj.get("footprint_geometry"), dict)
        assert isinstance(obj.get("surface_geometry"), dict)
        assert isinstance(obj.get("sphere_consistency"), dict)
        assert isinstance(obj.get("damage_metrics"), dict)
        assert "circularity" in (obj.get("footprint_geometry") or {})
        assert "radial_cv" in (obj.get("footprint_geometry") or {})
        assert "sphere_fit_rmse_mm" in (obj.get("surface_geometry") or {})
        assert "radial_height_rmse_mm" in (obj.get("sphere_consistency") or {})
        consistency = obj.get("sphere_consistency") or {}
        assert "surface_sphere_fit_rmse_norm" in consistency
        assert "surface_visible_cap_fraction" in consistency
        assert "surface_volume_fill_ratio_model" in consistency
        assert "flat_region_ratio" in (obj.get("damage_metrics") or {})
        assert isinstance(obj.get("superclass"), str)
        assert isinstance(obj.get("label"), str)
    object_candidates = payload.get("object_candidates")
    assert isinstance(object_candidates, list)
    if object_candidates:
        assert object_candidates[0].get("source_modality") == "derived_25d"


def test_25d_pipeline_emits_surface_sphere_fit_rmse(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    payload = _load_result(data_dir, take_id)
    output_dir = data_dir / "processed" / take_id

    assert payload.get("status") == "ok"
    objects = payload.get("objects") or []
    assert len(objects) > 0

    emitted: list[float] = []
    for obj in objects:
        surface = obj.get("surface_geometry") if isinstance(obj.get("surface_geometry"), dict) else {}
        value = surface.get("sphere_fit_rmse_mm")
        if value is not None:
            emitted.append(float(value))
        assert "surface_geometry" in obj

    assert emitted, "expected at least one object with finite surface_sphere_fit_rmse_mm"
    assert all(np.isfinite(value) and value >= 0.0 for value in emitted)

    feature_vector_path = output_dir / "feature_vector.json"
    assert feature_vector_path.is_file()
    feature_vector = json.loads(feature_vector_path.read_text(encoding="utf-8"))
    fv_value = (feature_vector.get("features") or {}).get("surface_sphere_fit_rmse_mm")
    assert fv_value is not None
    assert np.isfinite(float(fv_value))

    from vision_3d_acquisition.api.feature_catalog import feature_definition_for_key
    from vision_3d_acquisition.api.feature_analytics import _extract_feature_values

    definition = feature_definition_for_key("surface_sphere_fit_rmse_mm")
    assert "surface_geometry.sphere_fit_rmse_mm" in definition.extraction_paths
    for obj in objects:
        extracted, sources = _extract_feature_values(obj)
        if "surface_sphere_fit_rmse_mm" in extracted:
            assert extracted["surface_sphere_fit_rmse_mm"] >= 0.0
            assert sources["surface_sphere_fit_rmse_mm"] == "pipeline_run"
        if "surface_sphere_fit_rmse_norm" in extracted:
            assert extracted["surface_sphere_fit_rmse_norm"] >= 0.0
            assert sources["surface_sphere_fit_rmse_norm"] == "pipeline_run"


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
    assert files.get("feature_runtime_summary") == "feature_runtime_summary.json"
    assert files.get("feature_studio_summary") == "feature_studio_summary.json"

    summary = payload.get("summary") or {}
    classification = payload.get("classification") or {}
    assert isinstance(summary.get("object_count"), int)
    assert isinstance(summary.get("superclass"), str)
    assert isinstance(summary.get("label"), str)
    assert isinstance(summary.get("feature_runtime_summary"), dict)
    assert isinstance(summary.get("feature_readiness"), dict)
    assert isinstance(classification.get("superclass"), str)
    assert isinstance(classification.get("label"), str)
    pipeline_info = payload.get("processing_pipeline", {}) or {}
    pipeline_family = pipeline_info.get("pipeline_family")
    if pipeline_family is not None:
        assert pipeline_family == "25d"
    else:
        assert str(pipeline_info.get("id") or "").endswith("_25d")
    assert isinstance(payload.get("object_candidates"), list)

    objects = payload.get("objects") or []
    if objects:
        first = objects[0]
        assert isinstance(first.get("feature_group_summaries"), list)
        assert isinstance(first.get("feature_warnings"), list)
        assert isinstance(first.get("feature_readiness"), dict)


def test_25d_diagnostics_artifacts_are_generated_additively(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}

    for artifact_id in ["measurement_diagnostics", "feature_vector", "feature_provenance", "quality_flags", "contour", "convex_hull", "fitted_ellipse", "principal_axes", "radial_profile"]:
        assert artifact_id in by_id
        assert by_id[artifact_id].get("stage_id") == "measurement_diagnostics"
        assert by_id[artifact_id].get("kind") == "json"

    processed_dir = data_dir / "processed" / take_id
    feature_vector = json.loads((processed_dir / "feature_vector.json").read_text(encoding="utf-8"))
    provenance = json.loads((processed_dir / "feature_provenance.json").read_text(encoding="utf-8"))
    quality_flags = json.loads((processed_dir / "quality_flags.json").read_text(encoding="utf-8"))
    assert isinstance(feature_vector.get("features"), dict)
    assert "valid_pixel_ratio" in feature_vector["features"]
    assert isinstance(provenance.get("equivalent_diameter_mm"), dict)
    assert "source_stage" in provenance["equivalent_diameter_mm"]
    assert isinstance(quality_flags.get("flags"), list)


def test_25d_classification_provenance_defaults_present(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
    cls = payload.get("classification") or {}
    assert cls.get("classifier_engine") == "mining_steel_ball_classification_25d"
    assert cls.get("rule_set_id") == "builtin_default"
    assert cls.get("rule_set_source") == "builtin_default"
    objects = cls.get("objects") or []
    if objects:
        assert "rule_path" in objects[0]


def test_25d_classification_can_use_external_rule_set(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    cfg_dir = tmp_path / "configs" / "classifiers"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "demo_rules.json"
    cfg_path.write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "demo_v1",
                "params": {"good_min_sphericity": 0.7},
            }
        ),
        encoding="utf-8",
    )
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    payload = run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={"classify_25d": {"classifier_rules_path": str(cfg_path)}},
    ).result_payload
    cls = payload.get("classification") or {}
    assert cls.get("rule_set_id") == "demo_rules"
    assert cls.get("rule_set_version") == "demo_v1"
    assert cls.get("rule_set_source") == "runtime_override"


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
            "reference_surface_region_mode": "full_height_x_band",
            "plane_fit_roi": {"enabled": True, "type": "full_height_x_band", "x": 80, "width": 90},
        },
    }
    run_ball_inspection_25d_flow(data_dir, take_id=take_id, stage_params=roi_stage_params)
    payload = _load_result(data_dir, take_id)
    artifacts = payload.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}
    plane_fit_debug = (by_id.get("plane_fit_debug", {}).get("metadata") or {}) if isinstance(by_id.get("plane_fit_debug"), dict) else {}
    assert plane_fit_debug.get("roi_enabled") is True
    assert plane_fit_debug.get("roi_type") == "full_height_x_band"
    assert plane_fit_debug.get("roi_x") == 80
    assert plane_fit_debug.get("roi_y") == 0
    assert plane_fit_debug.get("roi_width") == 90
    assert plane_fit_debug.get("roi_height") == int(frame.z_mm.shape[0])


def test_25d_result_payload_preserves_nested_stage_params(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    stage_params = {
        "detect_belt_plane": {
            "background_detection_strategy": "low_gradient_surface",
            "reference_surface_model": "constant_z",
            "belt_stripe_filter_enabled": False,
        },
        "remove_belt_segment_objects": {
            "min_height_mm": 9.5,
            "max_height_mm": 150.0,
        },
        "known_object_25d": {
            "enabled": True,
            "target_selection": "manual_component_id",
            "manual_component_id": 1,
            "known_width_mm": 40.0,
            "known_depth_mm": 40.0,
            "known_height_mm": 25.0,
            "tolerance_percent": 5.0,
            "apply_correction": False,
        },
    }

    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id, stage_params=stage_params).result_payload

    assert payload.get("status") == "ok"
    assert payload.get("stage_params") == stage_params
    recipe_snapshot = payload.get("recipe_snapshot")
    assert isinstance(recipe_snapshot, dict)
    assert recipe_snapshot.get("pipeline_id") == "mining_steel_ball_classification_25d"
    assert recipe_snapshot.get("strategy_branch") == "low_gradient_surface"
    assert recipe_snapshot.get("stage_params") == stage_params
    unit_trace = payload.get("processing_unit_trace")
    assert isinstance(unit_trace, dict)
    unit_results = unit_trace.get("unit_results")
    assert isinstance(unit_results, dict)
    assert unit_trace.get("trace_source") == "mixed"
    assert unit_trace.get("trace_precision") == "mixed"
    refinement = unit_results.get("detect_belt_plane.candidate_support_refinement")
    assert isinstance(refinement, dict)
    assert refinement.get("trace_source") == "runtime_unit_callbacks"
    assert refinement.get("trace_precision") == "unit_level"
    assert refinement.get("status") in {"completed", "warning", "skipped"}
    assert isinstance(refinement.get("duration_ms"), int)
    assert "selected_blob_cluster_refined_mask" in list(refinement.get("output_artifacts") or [])
    params_used = refinement.get("parameters_used")
    assert isinstance(params_used, dict)
    assert "blob_cluster_refine_by_mad" in params_used
    segmentation = unit_results.get("remove_belt_segment_objects.morphology_cleanup")
    assert isinstance(segmentation, dict)
    assert segmentation.get("trace_source") == "runtime_unit_callbacks"
    segmentation_params = segmentation.get("parameters_used")
    assert isinstance(segmentation_params, dict)
    assert segmentation_params.get("morphology_kernel") == 5
    diagnostics = unit_results.get("measurement_diagnostics.known_object_validation")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("trace_source") == "runtime_unit_callbacks"
    diagnostics_params = diagnostics.get("parameters_used")
    assert isinstance(diagnostics_params, dict)
    assert diagnostics_params.get("enabled") is True


def test_runtime_trace_expands_coverage_across_non_detect_stages(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_trace_coverage")

    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload

    trace = payload.get("processing_unit_trace")
    assert isinstance(trace, dict)
    summary = trace.get("trace_summary")
    assert isinstance(summary, dict)
    assert int(summary.get("runtime_traced_units") or 0) >= 35
    coverage_by_stage = summary.get("coverage_by_stage")
    assert isinstance(coverage_by_stage, dict)
    assert int(((coverage_by_stage.get("normalize_heights_to_plane") or {}).get("runtime_traced_units") or 0)) >= 5
    assert int(((coverage_by_stage.get("geometry") or {}).get("runtime_traced_units") or 0)) >= 5
    assert int(((coverage_by_stage.get("measurement_diagnostics") or {}).get("runtime_traced_units") or 0)) >= 4
    assert int(((coverage_by_stage.get("overlay") or {}).get("runtime_traced_units") or 0)) >= 4

    unit_results = trace.get("unit_results")
    assert isinstance(unit_results, dict)

    normalize = unit_results.get("normalize_heights_to_plane.height_above_belt")
    assert isinstance(normalize, dict)
    assert normalize.get("trace_source") == "runtime_unit_callbacks"
    assert "height_max_mm" in dict(normalize.get("metrics") or {})

    geometry = unit_results.get("geometry.ellipse_fitting")
    assert isinstance(geometry, dict)
    assert geometry.get("trace_source") == "runtime_unit_callbacks"
    assert "ellipse_fit_success_count" in dict(geometry.get("metrics") or {})

    measurement = unit_results.get("measurement.height_metrics")
    assert isinstance(measurement, dict)
    assert measurement.get("trace_source") == "runtime_unit_callbacks"
    assert "max_height_max_mm" in dict(measurement.get("metrics") or {})

    diagnostics = unit_results.get("measurement_diagnostics.feature_vector_generation")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("trace_source") == "runtime_unit_callbacks"
    assert "feature_count" in dict(diagnostics.get("metrics") or {})

    overlay = unit_results.get("overlay.classification_overlay")
    assert isinstance(overlay, dict)
    assert overlay.get("trace_source") == "runtime_unit_callbacks"
    assert "classification_overlay_object_count" in dict(overlay.get("metrics") or {})


def test_25d_result_payload_defaults_to_empty_stage_params_without_overrides(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")

    payload = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload

    assert payload.get("status") == "ok"
    assert payload.get("stage_params") == {}


def test_reference_method_metadata_and_alias_masks_are_emitted(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_reference_method")

    payload = run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={
            "detect_belt_plane": {
                "background_detection_strategy": "low_gradient_blob_height_clusters",
                "blob_component_mode": "height_aware",
                "blob_split_method": "disabled",
                "blob_component_use_smoothed_z": False,
                "blob_neighbor_z_tolerance_mm": 2.0,
                "reference_surface_model": "constant_z",
                "reference_suppression_mask_policy": "selected_support",
            },
        },
    ).result_payload

    by_id = {item.get("artifact_id"): item for item in (payload.get("artifacts") or []) if isinstance(item, dict)}
    for artifact_id in ("selected_reference_support_mask", "reference_model_inlier_mask", "reference_suppression_mask"):
        assert artifact_id in by_id

    plane_fit_meta = (by_id.get("plane_fit_debug", {}).get("metadata") or {}) if isinstance(by_id.get("plane_fit_debug"), dict) else {}
    reference_method = plane_fit_meta.get("reference_method") or {}
    assert reference_method.get("preset_id") == "blob_height_aware_constant_z"
    assert reference_method.get("support_selection_method") == "blob_height_clusters"
    assert reference_method.get("background_detection_strategy") == "low_gradient_blob_height_clusters"
    assert reference_method.get("blob_component_mode") == "height_aware"
    assert reference_method.get("blob_split_method") == "disabled"
    assert reference_method.get("reference_surface_model") == "constant_z"
    assert reference_method.get("reference_suppression_mask_policy") == "selected_support"


def test_reference_suppression_mask_policy_selected_support_uses_selected_support_mask(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_suppression_policy")

    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={
            "detect_belt_plane": {
                "background_detection_strategy": "low_gradient_blob_height_clusters",
                "blob_component_mode": "height_aware",
                "blob_split_method": "disabled",
                "reference_surface_model": "constant_z",
                "reference_suppression_mask_policy": "selected_support",
            },
        },
    )

    out = data_dir / "processed" / take_id
    selected = (out / "selected_reference_support_mask.png").read_bytes()
    suppression = (out / "reference_suppression_mask.png").read_bytes()
    assert suppression == selected


def _run_low_gradient_bg_and_stripes_flow(tmp_path: Path, *, session_id: str) -> tuple[Path, dict[str, Any]]:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id=session_id)

    payload = run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={"detect_belt_plane": {"background_detection_strategy": "low_gradient_bg_and_stripes"}},
    ).result_payload
    return data_dir / "processed" / take_id, payload


def test_low_gradient_bg_and_stripes_strategy_preserves_support_and_suppression_invariants(tmp_path: Path) -> None:
    out, payload = _run_low_gradient_bg_and_stripes_flow(
        tmp_path,
        session_id="synthetic_25d_bg_and_stripes",
    )

    assert payload.get("status") == "ok"
    belt_bg = _load_mask_png(out / "belt_bg_mask.png")
    stripes = _load_mask_png(out / "belt_stripes_mask.png")
    suppression = _load_mask_png(out / "surface_suppression_mask.png")
    model_support = _load_mask_png(out / "reference_model_support_mask.png")
    unknown = _load_mask_png(out / "unknown_low_gradient_mask.png")
    object_search_domain = _load_mask_png(out / "object_search_domain_mask.png")

    assert not np.any(belt_bg & stripes)
    assert np.array_equal(model_support, belt_bg)
    assert np.array_equal(suppression, belt_bg | stripes)
    assert not np.any(unknown & suppression)
    assert np.array_equal(object_search_domain, ~(belt_bg | stripes) & object_search_domain)

    stripe_debug = json.loads((out / "belt_stripe_filter_debug.json").read_text(encoding="utf-8"))
    assert stripe_debug.get("background_detection_strategy") == "low_gradient_bg_and_stripes"
    invariants = stripe_debug.get("invariants") or {}
    assert invariants.get("background_support_equals_belt_bg") is True
    assert invariants.get("surface_suppression_equals_bg_or_stripes") is True
    assert invariants.get("stripes_excluded_from_background_fit") is True
    assert int(invariants.get("baseline_above_source_pixel_count") or 0) == 0
    assert int(invariants.get("negative_altitude_pixel_count") or 0) == 0
    assert int(invariants.get("unknown_low_gradient_suppressed_pixel_count") or 0) == 0

    segmentation_debug = json.loads((out / "segmentation_debug.json").read_text(encoding="utf-8"))
    assert segmentation_debug.get("suppression_source") == "surface_suppression_mask"
    assert int(segmentation_debug.get("unknown_low_gradient_suppressed_pixel_count") or 0) == 0


def test_low_gradient_bg_and_stripes_runtime_trace_reflects_reused_units(tmp_path: Path) -> None:
    _, payload = _run_low_gradient_bg_and_stripes_flow(
        tmp_path,
        session_id="synthetic_25d_bg_and_stripes_trace",
    )

    unit_results = (payload.get("processing_unit_trace") or {}).get("unit_results") or {}
    # depth_plateaus and blob_components genuinely execute inline for this hybrid strategy
    # (it calls the same _detect_depth_plateaus/_summarize_low_gradient_blobs functions the
    # dedicated strategies use), so their trace status must not read "skipped".
    assert unit_results.get("detect_belt_plane.depth_plateaus", {}).get("status") == "completed"
    assert unit_results.get("detect_belt_plane.blob_components", {}).get("status") == "completed"
    # blob_splitting and fragment_merge are NOT performed by this branch (no height-based
    # splitting or weak-boundary merge occurs) — must remain "skipped", not be over-corrected.
    assert unit_results.get("detect_belt_plane.blob_splitting", {}).get("status") == "skipped"
    assert unit_results.get("detect_belt_plane.fragment_merge", {}).get("status") == "skipped"


def test_low_gradient_bg_and_stripes_debug_counts_match_masks(tmp_path: Path) -> None:
    out, payload = _run_low_gradient_bg_and_stripes_flow(
        tmp_path,
        session_id="synthetic_25d_bg_and_stripes_debug_counts",
    )

    assert payload.get("status") == "ok"
    belt_bg = _load_mask_png(out / "belt_bg_mask.png")
    stripes = _load_mask_png(out / "belt_stripes_mask.png")
    unknown = _load_mask_png(out / "unknown_low_gradient_mask.png")
    suppression = _load_mask_png(out / "surface_suppression_mask.png")

    stripe_debug = json.loads((out / "belt_stripe_filter_debug.json").read_text(encoding="utf-8"))
    surface_roles = stripe_debug.get("surface_roles") or {}
    invariants = stripe_debug.get("invariants") or {}

    assert int(surface_roles.get("belt_bg_pixels") or 0) == int(np.count_nonzero(belt_bg))
    assert int(surface_roles.get("belt_stripe_pixels") or 0) == int(np.count_nonzero(stripes))
    assert int(surface_roles.get("unknown_low_gradient_pixels") or 0) == int(np.count_nonzero(unknown))
    assert int(surface_roles.get("suppression_pixels") or 0) == int(np.count_nonzero(suppression))
    assert int(surface_roles.get("bg_stripe_overlap_pixels") or 0) == int(np.count_nonzero(belt_bg & stripes))
    assert int(invariants.get("unknown_low_gradient_suppressed_pixel_count") or 0) == int(
        np.count_nonzero(unknown & suppression)
    )
    assert int(invariants.get("raw_negative_altitude_pixel_count") or 0) >= int(
        invariants.get("negative_altitude_pixel_count") or 0
    )


def test_low_gradient_bg_and_stripes_preserves_reference_support_and_objects(tmp_path: Path) -> None:
    out, payload = _run_low_gradient_bg_and_stripes_flow(
        tmp_path,
        session_id="synthetic_25d_bg_and_stripes_support",
    )

    assert payload.get("status") == "ok"
    belt_bg = _load_mask_png(out / "belt_bg_mask.png")
    stripes = _load_mask_png(out / "belt_stripes_mask.png")
    stripe_filtered = _load_mask_png(out / "stripe_filtered_reference_support_mask.png")
    selected_support = _load_mask_png(out / "selected_reference_support_mask.png")
    stripe_removed = _load_mask_png(out / "support_removed_by_stripe_filter.png")
    model_support = _load_mask_png(out / "reference_model_support_mask.png")
    suppression = _load_mask_png(out / "reference_suppression_mask.png")
    final_objects = _load_mask_png(out / "final_object_mask.png")

    assert np.array_equal(stripe_removed, selected_support & (~stripe_filtered))
    assert np.array_equal(selected_support, stripe_filtered)
    assert np.array_equal(model_support, belt_bg)
    assert np.array_equal(suppression, belt_bg | stripes)
    assert not np.any(final_objects & suppression)
    assert int(np.count_nonzero(final_objects)) > 0


def test_blob_height_cluster_strategy_emits_expected_debug_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_blob")

    payload = run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={"detect_belt_plane": {"background_detection_strategy": "low_gradient_blob_height_clusters"}},
    ).result_payload

    assert payload.get("status") == "ok"
    by_id = {item.get("artifact_id"): item for item in (payload.get("artifacts") or []) if isinstance(item, dict)}
    out = data_dir / "processed" / take_id
    for name in (
        "low_gradient_blob_components_overlay.png",
        "low_gradient_blob_id_mask.png",
        "low_gradient_blob_summary.json",
        "height_border_strength.png",
        "height_border_cut_mask.png",
        "height_border_fragments_overlay.png",
        "height_border_fragments_mask.png",
        "height_border_split_debug.json",
        "fragment_merge_debug.json",
        "height_split_blob_fragments_overlay.png",
        "height_split_blob_fragments_mask.png",
        "height_split_blob_id_mask.png",
        "height_split_debug.json",
        "height_consistent_blob_summary.json",
        "blob_height_clusters.json",
        "blob_cluster_score_table.csv",
        "selected_blob_cluster_mask.png",
        "selected_blob_cluster_pre_refine_mask.png",
        "selected_blob_cluster_refined_mask.png",
        "support_removed_by_candidate_refinement.png",
        "support_added_by_candidate_refinement.png",
        "support_removed_by_candidate_refinement_overlay.png",
        "selected_blob_cluster_overlay.png",
        "rejected_blob_clusters_mask.png",
        "rejected_blob_clusters_overlay.png",
        "blob_cluster_selection_debug.json",
        "stripe_filtered_reference_support_mask.png",
        "support_removed_by_stripe_filter.png",
        "support_added_by_stripe_filter.png",
        "support_removed_by_stripe_filter_overlay.png",
        "reference_model_support_mask.png",
        "support_removed_by_model_residual.png",
        "support_added_by_model_expansion.png",
        "support_removed_by_model_residual_overlay.png",
        "support_removed_by_suppression_policy.png",
        "support_added_by_suppression_policy.png",
        "support_removed_by_suppression_policy_overlay.png",
        "selected_support_lineage.json",
        "component_formation_debug.json",
        "height_border_detection_debug.json",
        "candidate_support_refinement_debug.json",
        "final_support_debug.json",
        "support_loss_waterfall.json",
    ):
        assert (out / name).is_file(), name
    for artifact_id in (
        "component_formation_debug",
        "height_border_detection_debug",
        "candidate_support_refinement_debug",
        "final_support_debug",
        "support_loss_waterfall",
        "final_selected_support_mask",
        "reference_model",
        "plane_residual_heatmap",
    ):
        assert artifact_id in by_id
        assert by_id[artifact_id].get("stage_id") == "detect_belt_plane"
        metadata = (by_id[artifact_id].get("metadata") or {}) if isinstance(by_id[artifact_id], dict) else {}
        assert "substage_id" in metadata
        assert "role" in metadata
        assert "order_index" in metadata
    assert (by_id["selected_reference_support_mask"].get("metadata") or {}).get("role") == "logical_support"
    assert (by_id["reference_model_support_mask"].get("metadata") or {}).get("role") == "fit_support"
    assert (by_id["reference_suppression_mask"].get("metadata") or {}).get("role") == "suppression_mask"
    gradient_debug = json.loads((out / "gradient_debug.json").read_text(encoding="utf-8"))
    assert gradient_debug.get("strategy") == "low_gradient_blob_height_clusters"
    blob_rows = json.loads((out / "low_gradient_blob_summary.json").read_text(encoding="utf-8"))
    assert isinstance(blob_rows, list)
    lineage = json.loads((out / "selected_support_lineage.json").read_text(encoding="utf-8"))
    assert lineage.get("strategy") == "low_gradient_blob_height_clusters"
    assert [step.get("id") for step in lineage.get("steps") or []] == [
        "selected_blob_cluster",
        "candidate_refinement",
        "stripe_suppression",
        "reference_model_support",
        "suppression_mask",
    ]
    steps = {str(step.get("id")): step for step in lineage.get("steps") or []}
    model_step = steps["reference_model_support"]
    assert "added_pixels" in model_step
    assert "added_fraction" in model_step
    assert "net_change_pixels" in model_step
    assert "net_change_fraction" in model_step
    assert model_step.get("change_type") in {"shrink", "expand", "same", "skipped", "alias", "unknown", "mixed"}
    assert "largest_support_gain_step" in (lineage.get("summary") or {})
    assert "largest_support_gain_fraction" in (lineage.get("summary") or {})

    refined_mask = _load_mask_png(out / "selected_blob_cluster_refined_mask.png")
    selected_support_mask = _load_mask_png(out / "selected_reference_support_mask.png")
    stripe_removed_mask = _load_mask_png(out / "support_removed_by_stripe_filter.png")
    stripe_added_mask = _load_mask_png(out / "support_added_by_stripe_filter.png")
    assert np.array_equal(stripe_removed_mask, refined_mask & (~selected_support_mask))
    assert np.array_equal(stripe_added_mask, selected_support_mask & (~refined_mask))

    model_input_mask = _load_mask_png(out / "selected_reference_support_mask.png")
    model_output_mask = _load_mask_png(out / "reference_model_support_mask.png")
    model_removed_mask = _load_mask_png(out / "support_removed_by_model_residual.png")
    model_added_mask = _load_mask_png(out / "support_added_by_model_expansion.png")
    assert np.array_equal(model_removed_mask, model_input_mask & (~model_output_mask))
    assert np.array_equal(model_added_mask, model_output_mask & (~model_input_mask))
    assert int(np.count_nonzero(model_added_mask)) == int(model_step.get("added_pixels") or 0)
    assert int(np.count_nonzero(model_output_mask) - np.count_nonzero(model_input_mask)) == int(model_step.get("net_change_pixels") or 0)
    if blob_rows:
        assert "blob_id" in blob_rows[0]
        assert "median_z" in blob_rows[0]
        assert "overlap_with_height_gate" in blob_rows[0]
    fragment_rows = json.loads((out / "height_consistent_blob_summary.json").read_text(encoding="utf-8"))
    assert isinstance(fragment_rows, list)
    if fragment_rows:
        assert "fragment_id" in fragment_rows[0]
        assert "source_blob_id" in fragment_rows[0]
        assert "was_split_from_blob" in fragment_rows[0]
    split_debug = json.loads((out / "height_split_debug.json").read_text(encoding="utf-8"))
    assert split_debug.get("strategy") == "low_gradient_blob_height_clusters"
    assert split_debug.get("method") in {"disabled", "histogram_gap", "height_borders", "height_borders_then_histogram"}
    assert "blobs" in split_debug
    border_debug = json.loads((out / "height_border_split_debug.json").read_text(encoding="utf-8"))
    assert border_debug.get("strategy") == "low_gradient_blob_height_clusters"
    merge_debug = json.loads((out / "fragment_merge_debug.json").read_text(encoding="utf-8"))
    assert merge_debug.get("strategy") == "low_gradient_blob_height_clusters"
    support_loss_waterfall = json.loads((out / "support_loss_waterfall.json").read_text(encoding="utf-8"))
    assert isinstance(support_loss_waterfall, list)
    assert [row.get("row_id") for row in support_loss_waterfall[:4]] == [
        "valid_roi",
        "low_gradient_candidates",
        "height_gate_candidates",
        "component_candidates",
    ]
    fit_support_row = next(row for row in support_loss_waterfall if row.get("row_id") == "reference_model_fit_support")
    assert "percent_valid_roi" in fit_support_row
    assert fit_support_row.get("status") == "ok"
    waterfall_meta = (by_id["support_loss_waterfall"].get("metadata") or {}) if isinstance(by_id["support_loss_waterfall"], dict) else {}
    assert isinstance(waterfall_meta.get("rows"), list)
    final_support_debug = json.loads((out / "final_support_debug.json").read_text(encoding="utf-8"))
    assert "stripe_overlap_with_fit_support_px" in final_support_debug


def test_blob_height_cluster_strategy_can_record_fallback(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_blob_fallback")

    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={
            "detect_belt_plane": {
                "background_detection_strategy": "low_gradient_blob_height_clusters",
                "blob_cluster_min_total_pixels": 10_000_000,
                "blob_cluster_fallback_strategy": "low_gradient_depth_plateaus",
            },
        },
    )

    out = data_dir / "processed" / take_id
    gradient_debug = json.loads((out / "gradient_debug.json").read_text(encoding="utf-8"))
    assert gradient_debug.get("strategy") == "low_gradient_blob_height_clusters"
    assert gradient_debug.get("fallback_used") is True
    assert gradient_debug.get("fallback_strategy") == "low_gradient_depth_plateaus"
    split_debug = json.loads((out / "height_split_debug.json").read_text(encoding="utf-8"))
    assert split_debug.get("enabled") is True


def test_selected_support_lineage_records_skipped_candidate_refinement(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_lineage_skip")

    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={
            "detect_belt_plane": {
                "background_detection_strategy": "low_gradient_blob_height_clusters",
                "blob_cluster_refine_by_mad": False,
                "blob_cluster_refine_keep_border_support": False,
            },
        },
    )

    lineage = json.loads((data_dir / "processed" / take_id / "selected_support_lineage.json").read_text(encoding="utf-8"))
    candidate_step = next(step for step in lineage.get("steps") or [] if step.get("id") == "candidate_refinement")
    assert candidate_step.get("status") == "skipped"
    assert candidate_step.get("net_change_pixels") == 0
    assert candidate_step.get("change_type") in {"skipped", "alias"}


def _blob_component_test_arrays(
    *,
    z_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z_values, dtype=np.float32)
    valid_mask = np.ones_like(z, dtype=bool)
    low_grad_mask = np.ones_like(z, dtype=bool)
    gradient = np.zeros_like(z, dtype=np.float32)
    roi_mask = np.ones_like(z, dtype=bool)
    return z, valid_mask, low_grad_mask, gradient, roi_mask


def test_blob_component_mode_xy_only_uses_classic_connected_components() -> None:
    z = np.zeros((40, 80), dtype=np.float32)
    z[:, :40] = 100.0
    z[:, 40:] = 130.0
    z_arr, valid_mask, low_grad_mask, gradient, roi_mask = _blob_component_test_arrays(z_values=z)
    xy_result = _summarize_low_gradient_blobs(
        low_grad_mask=low_grad_mask,
        gradient=gradient,
        z=z_arr,
        valid_mask=valid_mask,
        roi_mask=roi_mask,
        height_gate_mask=valid_mask,
        min_component_area=1,
        component_mode="xy_only",
    )
    assert xy_result.get("component_mode") == "xy_only"
    assert int(np.max(xy_result["labels"])) == 1


def test_height_aware_components_separate_sharp_touching_regions() -> None:
    z = np.zeros((40, 80), dtype=np.float32)
    z[:, :40] = 100.0
    z[:, 40:] = 130.0
    z_arr, valid_mask, low_grad_mask, gradient, roi_mask = _blob_component_test_arrays(z_values=z)
    result = _summarize_low_gradient_blobs(
        low_grad_mask=low_grad_mask,
        gradient=gradient,
        z=z_arr,
        valid_mask=valid_mask,
        roi_mask=roi_mask,
        height_gate_mask=valid_mask,
        min_component_area=1,
        component_mode="height_aware",
        blob_neighbor_z_tolerance_mm=3.0,
        blob_component_min_area_px=1,
        blob_component_use_smoothed_z=False,
    )
    assert result.get("component_mode") == "height_aware"
    assert len(result.get("blobs") or []) == 2
    debug = result.get("connectivity_debug") or {}
    assert int(debug.get("rejected_neighbor_edges") or 0) >= 1


def test_height_aware_components_keep_smooth_sloped_belt_connected() -> None:
    z = np.zeros((20, 120), dtype=np.float32)
    for x in range(120):
        z[:, x] = 100.0 + 0.5 * x
    z_arr, valid_mask, low_grad_mask, gradient, roi_mask = _blob_component_test_arrays(z_values=z)
    result = _summarize_low_gradient_blobs(
        low_grad_mask=low_grad_mask,
        gradient=gradient,
        z=z_arr,
        valid_mask=valid_mask,
        roi_mask=roi_mask,
        height_gate_mask=valid_mask,
        min_component_area=1,
        component_mode="height_aware",
        blob_neighbor_z_tolerance_mm=3.0,
        blob_component_allow_gradual_slope=True,
        blob_component_max_local_slope_mm_per_px=3.0,
        blob_component_min_area_px=1,
    )
    assert len(result.get("blobs") or []) == 1


def test_height_aware_tolerance_controls_component_count_and_rejected_edges() -> None:
    z = np.zeros((20, 120), dtype=np.float32)
    for x in range(120):
        z[:, x] = 100.0 + 1.0 * x
    z_arr, valid_mask, low_grad_mask, gradient, roi_mask = _blob_component_test_arrays(z_values=z)
    loose = _summarize_low_gradient_blobs(
        low_grad_mask=low_grad_mask,
        gradient=gradient,
        z=z_arr,
        valid_mask=valid_mask,
        roi_mask=roi_mask,
        height_gate_mask=valid_mask,
        min_component_area=1,
        component_mode="height_aware",
        blob_neighbor_z_tolerance_mm=3.0,
        blob_component_min_area_px=1,
    )
    strict = _summarize_low_gradient_blobs(
        low_grad_mask=low_grad_mask,
        gradient=gradient,
        z=z_arr,
        valid_mask=valid_mask,
        roi_mask=roi_mask,
        height_gate_mask=valid_mask,
        min_component_area=1,
        component_mode="height_aware",
        blob_neighbor_z_tolerance_mm=0.5,
        blob_component_min_area_px=1,
    )
    loose_debug = loose.get("connectivity_debug") or {}
    strict_debug = strict.get("connectivity_debug") or {}
    assert len(strict.get("blobs") or []) >= len(loose.get("blobs") or [])
    assert int(strict_debug.get("rejected_neighbor_edges") or 0) >= int(loose_debug.get("rejected_neighbor_edges") or 0)


def test_height_aware_disabled_split_emits_expected_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_blob_height_aware")
    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={
            "detect_belt_plane": {
                "background_detection_strategy": "low_gradient_blob_height_clusters",
                "blob_component_mode": "height_aware",
                "blob_split_method": "disabled",
                "blob_split_by_height_enabled": False,
            },
        },
    )
    out = data_dir / "processed" / take_id
    for name in (
        "height_aware_blob_components_overlay.png",
        "height_aware_blob_id_mask.png",
        "height_aware_connectivity_rejected_edges.png",
        "height_aware_connectivity_debug.json",
        "low_gradient_blob_components_overlay.png",
        "low_gradient_blob_summary.json",
    ):
        assert (out / name).is_file(), name
    gradient_debug = json.loads((out / "gradient_debug.json").read_text(encoding="utf-8"))
    assert gradient_debug.get("component_mode") == "height_aware"
    assert gradient_debug.get("split_method") == "disabled"
    connectivity_debug = json.loads((out / "height_aware_connectivity_debug.json").read_text(encoding="utf-8"))
    assert connectivity_debug.get("parameters", {}).get("blob_component_mode") == "height_aware"


def test_reference_audit_includes_height_aware_artifacts(tmp_path: Path) -> None:
    from vision_3d_acquisition.debug.reference_audit import build_reference_audit_bundle

    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_blob_audit")
    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={
            "detect_belt_plane": {
                "background_detection_strategy": "low_gradient_blob_height_clusters",
                "blob_component_mode": "height_aware",
                "blob_split_method": "disabled",
                "blob_split_by_height_enabled": False,
            },
        },
    )
    bundle = build_reference_audit_bundle(data_dir, take_id=take_id)
    manifest = bundle.manifest
    assert "height_aware_blob_components_overlay" in manifest.get("artifacts_found", [])
    assert "height_aware_connectivity_debug" in manifest.get("artifacts_found", [])
    assert bundle.output_dir.joinpath("04_blob_clusters/height_aware_connectivity_debug.json").is_file()


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
    assert isinstance(explanation.get("feature_runtime_summary"), dict)
    assert isinstance(explanation.get("feature_readiness"), dict)
    assert isinstance(explanation.get("objects"), list)
    if explanation.get("objects"):
        first = explanation["objects"][0]
        assert isinstance(first.get("feature_group_summaries"), list)
        assert isinstance(first.get("feature_warnings"), list)
        assert isinstance(first.get("feature_readiness"), dict)
    assert (payload.get("files") or {}).get("classification_explanation") == "classification_explanation.json"
    assert "metric_explanation" in by_id
    metric_art = by_id["metric_explanation"]
    assert metric_art.get("kind") == "json"
    assert metric_art.get("stage_id") == "classification"
    assert metric_art.get("path") == "metric_explanation.json"
    assert (payload.get("files") or {}).get("metric_explanation") == "metric_explanation.json"
    assert "feature_runtime_summary" in by_id
    assert "feature_studio_summary" in by_id
    assert (payload.get("files") or {}).get("feature_runtime_summary") == "feature_runtime_summary.json"
    assert (payload.get("files") or {}).get("feature_studio_summary") == "feature_studio_summary.json"


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


def test_classification_explanation_includes_sphere_consistency_rules(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="synthetic_25d_demo")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    explanation = json.loads((data_dir / "processed" / take_id / "classification_explanation.json").read_text(encoding="utf-8"))
    explained_objects = explanation.get("objects") or []
    assert explained_objects
    rule_ids = {str(rule.get("rule_id") or "") for obj in explained_objects for rule in (obj.get("rules") or [])}
    assert "consistency.sphere_rmse_norm" in rule_ids
    assert "consistency.sphere_radius_error" in rule_ids
    assert "consistency.volume_fill_ratio" in rule_ids

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
        "footprint_geometry": {"radial_cv": 0.05},
        "surface_geometry": {"sphere_fit_rmse_mm": 2.5},
        "sphere_consistency": {"radial_height_rmse_mm": 2.0, "surface_completeness_ratio": 0.82},
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


def test_load_heightmap_frame_excludes_nonpositive_z(tmp_path: Path) -> None:
    repo_take = Path(__file__).resolve().parents[1] / "data" / "incoming" / "2026-05-25T145918_022"
    if not (repo_take / "height16.tif").is_file():
        pytest.skip("fixture take not available")

    metadata = json.loads((repo_take / "metadata.json").read_text(encoding="utf-8"))
    from vision_3d_acquisition.vision_core.pipelines.stages_25d import _load_heightmap_frame

    frame = _load_heightmap_frame(
        repo_take / "height16.tif",
        repo_take / "reflectance.png",
        metadata,
    )
    valid_z = frame.z_mm[frame.valid_mask]
    assert valid_z.size > 0
    assert float(np.min(valid_z)) > 0.0
    assert int(np.count_nonzero(frame.valid_mask & (frame.z_mm <= 0.0))) == 0

    stale_npz = tmp_path / "heightmap_frame.npz"
    save_heightmap_npz(
        HeightmapFrame(
            z_mm=frame.z_mm,
            valid_mask=np.isfinite(frame.z_mm),
            reflectance=None,
            x_resolution_mm=frame.x_resolution_mm,
            y_resolution_mm=frame.y_resolution_mm,
            origin_x_mm=frame.origin_x_mm,
            origin_y_mm=frame.origin_y_mm,
            coordinate_system=frame.coordinate_system,
        ),
        stale_npz,
    )
    reloaded = _load_heightmap_frame(stale_npz, None, metadata)
    assert int(np.count_nonzero(reloaded.valid_mask & (reloaded.z_mm <= 0.0))) == 0
