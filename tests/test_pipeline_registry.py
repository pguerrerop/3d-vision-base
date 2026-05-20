from __future__ import annotations

from vision_3d_acquisition.pipelines.registry import (
    build_stage_outputs,
    get_pipeline,
    get_processing_unit,
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
