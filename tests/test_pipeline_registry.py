from __future__ import annotations

from vision_3d_acquisition.pipelines.registry import (
    build_stage_outputs,
    get_pipeline,
    get_processing_unit,
    list_processing_unit_definitions,
    list_pipelines,
    list_processing_units,
)


def test_pipeline_registry_lists_current_and_future_pipelines() -> None:
    pipelines = list_pipelines()
    ids = {pipeline["id"] for pipeline in pipelines}

    assert "3d_ball_inspection" in ids
    assert "2d_3d_fusion" in ids
    fusion = get_pipeline("2d_3d_fusion")
    assert fusion is not None
    assert fusion["implemented"] is False
    assert fusion["required_modalities"] == ["point_cloud", "rgb"]


def test_stage_outputs_group_artifacts_by_stage() -> None:
    outputs = build_stage_outputs(
        {
            "input_preview": "input_point_cloud_preview.png",
            "debug_foreground": "debug_foreground.png",
            "debug_clusters": "debug_clusters.png",
            "overlay": "overlay.png",
        }
    )

    segmentation = next(output for output in outputs if output["stage"] == "segmentation")
    classification = next(output for output in outputs if output["stage"] == "classification")
    fusion = next(output for output in outputs if output["stage"] == "fusion")

    assert segmentation["artifacts"]["foreground"] == "debug_foreground.png"
    assert segmentation["artifacts"]["clusters"] == "debug_clusters.png"
    assert classification["artifacts"]["overlay"] == "overlay.png"
    assert fusion["implemented"] is False


def test_processing_unit_registry_exposes_metadata() -> None:
    units = list_processing_units("3d_ball_inspection")
    classification = get_processing_unit("classification", "3d_ball_inspection")
    assert units
    assert classification is not None
    assert classification["display_name"] == "Ball classification"
    assert "overlay" in classification["produced_artifact_kinds"]
    assert classification["object_outputs"] is True


def test_detect_reference_processing_unit_registry_exposes_contract_definitions() -> None:
    units = list_processing_unit_definitions("mining_steel_ball_classification_25d")
    assert units
    ids = [str(unit["id"]) for unit in units]
    assert ids[0] == "input"
    assert "detect_belt_plane" in ids
    assert "normalize_heights_to_plane" in ids
    assert "remove_belt_segment_objects" in ids
    assert "geometry" in ids
    assert "measurement" in ids
    assert "measurement_diagnostics" in ids
    assert "classification" in ids
    assert "overlay" in ids
    assert "detect_belt_plane.depth_gradient" in ids
    assert "detect_belt_plane.depth_plateaus" in ids
    assert "detect_belt_plane.blob_components" in ids
    assert "detect_belt_plane.blob_splitting" in ids
    assert "detect_belt_plane.fragment_merge" in ids
    assert "detect_belt_plane.candidate_support_refinement" in ids
    assert "detect_belt_plane.stripe_filter" in ids
    assert "detect_belt_plane.reference_model_fit" in ids
    assert "detect_belt_plane.final_support" in ids

    assert len(ids) == len(set(ids))
    by_id = {str(unit["id"]): unit for unit in units}
    # Stage-root units (input, detect_belt_plane, normalize_heights_to_plane, ...) legitimately
    # have parent_id=None; only substages carry a parent. Every parent that IS set must resolve.
    for unit in units:
        parent_id = unit.get("parent_id")
        if parent_id is not None:
            assert parent_id in by_id, f"{unit['id']} references unknown parent {parent_id}"
    for unit in units:
        param_ids = [str(item["id"]) for item in unit.get("parameters") or []]
        assert len(param_ids) == len(set(param_ids))
    roi_unit = by_id["detect_belt_plane.roi"]
    roi_params = {str(item["id"]): item for item in roi_unit.get("parameters") or []}
    assert "reference_surface_region" in roi_params
    assert roi_params["reference_surface_region"]["type"] == "roi"
    segmentation_root = by_id["remove_belt_segment_objects"]
    segmentation_params = {str(item["id"]) for item in segmentation_root.get("parameters") or []}
    assert {"min_height_mm", "max_height_mm", "morphology_kernel", "min_component_area", "fill_holes"} <= segmentation_params
    diagnostics_root = by_id["measurement_diagnostics"]
    diagnostics_params = {str(item["id"]) for item in diagnostics_root.get("parameters") or []}
    assert {"enabled", "target_selection", "manual_component_id", "known_width_mm", "apply_correction"} <= diagnostics_params


def test_2d_segmentation_stage_exposes_parameter_schema() -> None:
    pipeline = get_pipeline("mining_steel_ball_classification_2d")
    assert pipeline is not None
    segmentation = next(stage for stage in pipeline["stages"] if stage["id"] == "segmentation")
    schema = segmentation.get("parameter_schema")
    assert isinstance(schema, dict)
    fields = schema.get("fields")
    assert isinstance(fields, dict)
    for key in ("threshold", "auto_threshold", "morph_op", "erode_kernel_size", "dilate_iterations", "min_area_px"):
        assert key in fields


def test_mining_rgb_pipeline_does_not_require_reflectance() -> None:
    pipeline = get_pipeline("mining_steel_ball_classification_2d")
    assert pipeline is not None
    assert pipeline["required_modalities"] == ["rgb"]
    for stage in pipeline["stages"]:
        if stage["id"] in {"segmentation", "detection", "measurement", "classification"}:
            assert stage["required_modalities"] == ["rgb"]


def test_25d_pipeline_stage_order_exposes_plane_qa_split() -> None:
    pipeline = get_pipeline("mining_steel_ball_classification_25d")
    assert pipeline is not None
    stage_ids = [stage["id"] for stage in pipeline["stages"]]
    assert stage_ids[:4] == [
        "input",
        "detect_belt_plane",
        "normalize_heights_to_plane",
        "remove_belt_segment_objects",
    ]
    assert pipeline["composition"]["execution_order"][:4] == stage_ids[:4]


def test_25d_pipeline_registry_exposes_first_class_parameter_schemas() -> None:
    pipeline = get_pipeline("mining_steel_ball_classification_25d")
    assert pipeline is not None
    by_id = {stage["id"]: stage for stage in pipeline["stages"]}

    detect_schema = by_id["detect_belt_plane"].get("parameter_schema")
    assert isinstance(detect_schema, dict)
    assert detect_schema.get("runtime_stage_params_key") == "detect_belt_plane"
    assert detect_schema.get("payload_path") == "stage_params.detect_belt_plane"
    assert [group.get("label") for group in detect_schema.get("groups", [])] == [
        "Reference surface",
        "Advanced reference tuning",
        "Belt stripe suppression",
    ]
    detect_fields = detect_schema.get("fields")
    assert isinstance(detect_fields, dict)
    assert detect_fields["background_detection_strategy"]["default"] == "low_gradient_surface"
    assert detect_fields["background_detection_strategy"]["enum"] == [
        "low_gradient_surface",
        "low_gradient_depth_plateaus",
        "low_gradient_bg_and_stripes",
        "low_gradient_blob_height_clusters",
        "nearest_percentile",
        "farthest_percentile",
        "automatic",
    ]
    assert detect_fields["reference_surface_model"]["enum"] == ["auto", "plane", "constant_z"]
    assert detect_fields["reference_suppression_mask_policy"]["enum"] == [
        "auto",
        "selected_support",
        "expanded_support",
        "final_model_inliers",
    ]
    assert detect_fields["plane_fit_min_inlier_ratio"]["default"] == 0.35
    assert detect_fields["plane_fit_residual_threshold_mm"]["default"] == 1.25
    assert detect_fields["plane_background_residual_tolerance_mm"]["default"] == 2.5
    assert detect_fields["belt_stripe_filter_scope"]["enum"] == ["global", "bg_plateau"]
    assert detect_fields["belt_stripe_filter_window_mm"]["default"] == 30.0
    assert detect_fields["belt_stripe_filter_object_kernel_mm"]["default"] == 100.0
    assert detect_fields["blob_cluster_height_gap_mode"]["enum"] == ["fixed", "adaptive"]
    assert detect_fields["blob_cluster_fallback_strategy"]["enum"] == [
        "low_gradient_depth_plateaus",
        "low_gradient_surface",
        "constant_z",
        "fail",
    ]
    assert detect_fields["blob_split_by_height_enabled"]["default"] is True
    assert detect_fields["blob_split_method"]["default"] == "height_borders"
    assert detect_fields["blob_split_method"]["enum"] == [
        "disabled",
        "histogram_gap",
        "height_borders",
        "height_borders_then_histogram",
    ]
    assert detect_fields["blob_split_mode"]["enum"] == ["histogram_gap"]
    assert detect_fields["blob_split_hist_bins"]["default"] == 48
    assert detect_fields["blob_split_min_band_fraction"]["default"] == 0.08
    assert detect_fields["blob_split_close_kernel"]["default"] == 3
    assert detect_fields["blob_split_height_border_mode"]["default"] == "morphological_gradient"
    assert detect_fields["blob_split_height_border_threshold_mode"]["enum"] == ["percentile", "fixed", "otsu"]
    assert detect_fields["blob_split_merge_max_boundary_strength_mm"]["default"] == 6.0

    segmentation_schema = by_id["remove_belt_segment_objects"].get("parameter_schema")
    assert isinstance(segmentation_schema, dict)
    assert segmentation_schema.get("runtime_stage_params_key") == "remove_belt_segment_objects"
    segmentation_fields = segmentation_schema.get("fields")
    assert isinstance(segmentation_fields, dict)
    assert segmentation_fields["min_height_mm"]["default"] == 8.0
    assert segmentation_fields["max_height_mm"]["default"] is None
    assert segmentation_fields["morphology_kernel"]["default"] == 5
    assert segmentation_fields["min_component_area"]["default"] == 120
    assert segmentation_fields["fill_holes"]["default"] is True

    known_schema = by_id["measurement_diagnostics"].get("parameter_schema")
    assert isinstance(known_schema, dict)
    assert known_schema.get("runtime_stage_params_key") == "known_object_25d"
    assert known_schema.get("payload_path") == "stage_params.known_object_25d"
    known_fields = known_schema.get("fields")
    assert isinstance(known_fields, dict)
    assert known_fields["enabled"]["default"] is False
    assert known_fields["target_selection"]["default"] == "largest_component"
    assert known_fields["target_selection"]["enum"] == ["largest_component", "manual_component_id"]
    assert known_fields["manual_component_id"]["default"] is None
    assert known_fields["tolerance_percent"]["default"] == 5.0
    assert known_fields["apply_correction"]["default"] is True
