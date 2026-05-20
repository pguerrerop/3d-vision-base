from __future__ import annotations

from vision_3d_acquisition.processes.models import ProcessTemplate, StepDefinition


def mining_steel_ball_template() -> ProcessTemplate:
    return ProcessTemplate(
        id="mining_steel_ball_classification_2d_reflectance_mvp",
        name="Mining Steel Ball Classification (RGB/2D MVP)",
        category="image_2d",
        description="Guided RGB-first 2D pipeline for steel-ball inspection with grayscale conversion, lighting normalization, thresholding, morphology cleanup, contour extraction, ellipse fitting, and classification.",
        supported_input_types=["rgb_image", "grayscale_image", "reflectance_image"],
        default_parameters={
            "crop_roi": {"enabled": False, "x": 0, "y": 0, "width": 640, "height": 480},
            "rgb_to_gray": {"method": "luminance"},
            "normalize_lighting": {
                "method": "clahe",
                "clip_limit": 2.4,
                "tile_grid_size": 10,
                "background_blur_kernel": 41,
            },
            "threshold": {
                "mode": "fixed",
                "value": 125,
                "adaptive_block_size": 31,
                "adaptive_c": 3,
                "blur_kernel": 5,
                "invert": False,
                "roi_enabled": False,
                "roi_type": "rectangle",
                "roi_x": 0,
                "roi_y": 0,
                "roi_width": 1920,
                "roi_height": 1080,
                "roi_polygon_points": [],
            },
            "morphology": {
                "operation": "open_close",
                "open_kernel": 3,
                "close_kernel": 5,
                "iterations": 1,
                "fill_holes": True,
                "cleanup_min_area": 120,
                "cleanup_max_area": None,
                "cleanup_min_width": 0,
                "cleanup_min_height": 0,
                "cleanup_max_aspect_ratio": 0.0,
                "cleanup_border_reject": False,
                "cleanup_keep_largest_n": 0,
                "overlay_alpha": 0.35,
                "overlay_color_r": 0,
                "overlay_color_g": 255,
                "overlay_color_b": 0,
            },
            "blob_detection": {
                "min_area": 120.0,
                "max_area": 250000.0,
                "min_width": 0,
                "min_height": 0,
                "max_aspect_ratio": 0.0,
                "min_circularity": 0.45,
                "reject_border_touching": False,
                "keep_largest_n": 0,
                "max_objects": 100,
            },
            "ellipse_fitting": {
                "enabled": True,
                "min_fit_points": 8,
                "max_fit_rmse": 8.0,
                "max_eccentricity": 0.9,
                "min_fill_ratio": 0.55,
                "reject_border_touching": False,
                "fit_error_threshold": 0.22,
            },
            "classification": {
                "min_area": 120.0,
                "max_area": 250000.0,
                "min_circularity_for_ball": 0.82,
                "min_confidence": 0.45,
                "spherical_enough_threshold": 0.78,
                "diameter_min": 10.0,
                "diameter_max": 300.0,
            },
        },
        ui_metadata={
            "application": "Mining Steel Ball Classification",
            "guided_steps": ["Choose Application", "Choose Input Type", "Create Pipeline"],
            "default_input_type": "rgb_image",
            "family": "2d_inspection",
        },
        steps=[
            StepDefinition(step_id="input", title="Input RGB image", namespace="image_2d.preprocess", default_algorithm_key="image_2d.preprocess.input", output_artifact_types=["source_rgb_image"]),
            StepDefinition(step_id="crop_roi", title="ROI selection / crop", namespace="image_2d.preprocess", default_algorithm_key="image_2d.preprocess.crop_roi", output_artifact_types=["roi_image", "roi_metadata"], overlay_types=["bbox"], params_schema={"enabled": {"type": "boolean"}, "x": {"type": "integer", "minimum": 0}, "y": {"type": "integer", "minimum": 0}, "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1}}),
            StepDefinition(step_id="rgb_to_gray", title="RGB to grayscale", namespace="image_2d.preprocess", default_algorithm_key="image_2d.preprocess.rgb_to_gray", output_artifact_types=["grayscale_image"], params_schema={"method": {"type": "string", "enum": ["luminance", "red", "green", "blue", "max_channel"]}}),
            StepDefinition(step_id="normalize_lighting", title="Lighting normalization", namespace="image_2d.preprocess", default_algorithm_key="image_2d.preprocess.normalize_lighting", output_artifact_types=["normalized_grayscale"], params_schema={"method": {"type": "string", "enum": ["none", "clahe", "background_subtract"]}, "clip_limit": {"type": "number", "minimum": 0.1, "maximum": 20.0}, "tile_grid_size": {"type": "integer", "minimum": 2, "maximum": 64}, "background_blur_kernel": {"type": "integer", "minimum": 3, "maximum": 201}}),
            StepDefinition(step_id="threshold", title="Threshold", namespace="image_2d.segment", default_algorithm_key="image_2d.segment.threshold", output_artifact_types=["threshold_mask"], params_schema={"mode": {"type": "string", "enum": ["fixed", "otsu", "adaptive"]}, "value": {"type": "number", "minimum": 0, "maximum": 255}, "invert": {"type": "boolean"}, "adaptive_block_size": {"type": "integer", "minimum": 3, "maximum": 255}, "adaptive_c": {"type": "number", "minimum": -255, "maximum": 255}, "blur_kernel": {"type": "integer", "minimum": 1, "maximum": 41}, "roi_enabled": {"type": "boolean"}, "roi_type": {"type": "string", "enum": ["rectangle", "polygon"]}, "roi_x": {"type": "integer", "minimum": 0}, "roi_y": {"type": "integer", "minimum": 0}, "roi_width": {"type": "integer", "minimum": 1}, "roi_height": {"type": "integer", "minimum": 1}, "roi_polygon_points": {"type": "array"}}),
            StepDefinition(step_id="morphology", title="Morphology cleanup", namespace="image_2d.segment", default_algorithm_key="image_2d.segment.morphology", output_artifact_types=["cleaned_mask", "overlay_image", "morphology_metrics", "morphology_debug_json", "rejected_components_overlay"], params_schema={"operation": {"type": "string", "enum": ["open_close", "close_open", "open_only", "close_only", "erode", "dilate", "none"]}, "open_kernel": {"type": "integer", "minimum": 1, "maximum": 31}, "close_kernel": {"type": "integer", "minimum": 1, "maximum": 31}, "iterations": {"type": "integer", "minimum": 1, "maximum": 20}, "fill_holes": {"type": "boolean"}, "cleanup_min_area": {"type": "integer", "minimum": 0, "maximum": 10000000}, "cleanup_max_area": {"type": "integer", "minimum": 0, "maximum": 100000000}, "cleanup_min_width": {"type": "integer", "minimum": 0, "maximum": 100000}, "cleanup_min_height": {"type": "integer", "minimum": 0, "maximum": 100000}, "cleanup_max_aspect_ratio": {"type": "number", "minimum": 0.0, "maximum": 1000.0}, "cleanup_border_reject": {"type": "boolean"}, "cleanup_keep_largest_n": {"type": "integer", "minimum": 0, "maximum": 100000}, "overlay_alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}, "overlay_color_r": {"type": "integer", "minimum": 0, "maximum": 255}, "overlay_color_g": {"type": "integer", "minimum": 0, "maximum": 255}, "overlay_color_b": {"type": "integer", "minimum": 0, "maximum": 255}}),
            StepDefinition(step_id="blob_detection", title="Blob/contour detection", namespace="image_2d.detect", default_algorithm_key="image_2d.detect.blob_contours", output_artifact_types=["contour_blob_artifact", "blob_debug_overlay", "blob_labels", "blob_contours", "blob_metrics", "blob_rejected"], overlay_types=["polyline", "bbox", "centroid"], params_schema={"min_area": {"type": "number", "minimum": 0}, "max_area": {"type": "number", "minimum": 0}, "min_width": {"type": "integer", "minimum": 0}, "min_height": {"type": "integer", "minimum": 0}, "max_aspect_ratio": {"type": "number", "minimum": 0.0}, "min_circularity": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "reject_border_touching": {"type": "boolean"}, "keep_largest_n": {"type": "integer", "minimum": 0}, "max_objects": {"type": "integer", "minimum": 1, "maximum": 10000}}),
            StepDefinition(step_id="ellipse_fitting", title="Ellipse/circle fitting", namespace="image_2d.measure", default_algorithm_key="image_2d.measure.ellipse_fit", output_artifact_types=["ellipse_overlay", "ellipse_metrics", "ellipse_summary", "ellipse_debug_overlay", "ellipse_fit_artifact"], overlay_types=["ellipse", "polyline", "text"], params_schema={"enabled": {"type": "boolean"}, "min_fit_points": {"type": "integer", "minimum": 5, "maximum": 1000}, "max_fit_rmse": {"type": "number", "minimum": 0.0, "maximum": 1000.0}, "max_eccentricity": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "min_fill_ratio": {"type": "number", "minimum": 0.0, "maximum": 2.0}, "reject_border_touching": {"type": "boolean"}, "fit_error_threshold": {"type": "number", "minimum": 0.0, "maximum": 2.0}}),
            StepDefinition(step_id="metrics", title="Metrics extraction", namespace="image_2d.measure", default_algorithm_key="image_2d.measure.metrics", output_artifact_types=["measurement_table_artifact"]),
            StepDefinition(step_id="classification", title="Classification", namespace="image_2d.classify", default_algorithm_key="image_2d.classify.ball_classifier", output_artifact_types=["classification_result_artifact"], overlay_types=["text"], params_schema={"min_area": {"type": "number", "minimum": 0}, "max_area": {"type": "number", "minimum": 0}, "min_circularity_for_ball": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "spherical_enough_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "diameter_min": {"type": "number", "minimum": 0.0}, "diameter_max": {"type": "number", "minimum": 0.0}}),
            StepDefinition(step_id="overlay", title="Overlay generation", namespace="image_2d.classify", default_algorithm_key="image_2d.classify.overlay", output_artifact_types=["overlay_summary_artifact"], overlay_types=["polyline", "ellipse", "centroid", "text"]),
            StepDefinition(step_id="summary", title="Result summary", namespace="image_2d.classify", default_algorithm_key="image_2d.classify.summary", output_artifact_types=["result_summary_artifact"]),
        ],
    )


def generic_2d_blob_template() -> ProcessTemplate:
    template = mining_steel_ball_template()
    return template.model_copy(update={
        "id": "generic_2d_blob_inspection",
        "name": "Generic 2D Blob Inspection",
        "description": "Basic 2D threshold/morphology/blob pipeline.",
        "ui_metadata": {"application": "Generic 2D Blob Inspection", "family": "2d_inspection", "default_input_type": "rgb_image"},
    })


def generic_3d_segmentation_placeholder() -> ProcessTemplate:
    return ProcessTemplate(
        id="generic_3d_segmentation_placeholder",
        name="Generic 3D Segmentation (placeholder)",
        category="image_3d",
        description="Placeholder template for future guided 3D segmentation workflows.",
        supported_input_types=["point_cloud"],
        steps=[],
        default_parameters={},
        ui_metadata={"placeholder": True},
    )


def custom_pipeline_placeholder() -> ProcessTemplate:
    return ProcessTemplate(
        id="custom_pipeline_placeholder",
        name="Custom Pipeline (placeholder)",
        category="hybrid",
        description="Placeholder for future custom pipeline editor without graph workflows.",
        supported_input_types=["grayscale_image", "reflectance_image", "point_cloud", "rgb_image"],
        steps=[],
        default_parameters={},
        ui_metadata={"placeholder": True},
    )


def list_templates() -> list[ProcessTemplate]:
    return [
        mining_steel_ball_template(),
        generic_2d_blob_template(),
        generic_3d_segmentation_placeholder(),
        custom_pipeline_placeholder(),
    ]


def get_template(template_id: str) -> ProcessTemplate | None:
    for template in list_templates():
        if template.id == template_id:
            return template
    return None
