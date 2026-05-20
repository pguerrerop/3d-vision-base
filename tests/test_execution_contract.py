from __future__ import annotations

from vision_3d_acquisition.contracts.execution import build_pipeline_execution_trace
from vision_3d_acquisition.pipelines.registry import default_pipeline_info


def test_pipeline_execution_trace_serialization_supports_skipped_and_success() -> None:
    pipeline = default_pipeline_info("3d_ball_inspection")
    trace = build_pipeline_execution_trace(
        pipeline=pipeline,
        artifacts=[
            {"artifact_id": "foreground_clusters", "stage_id": "segmentation"},
            {"artifact_id": "classification_table", "stage_id": "classification"},
        ],
        profiling={
            "stages": [
                {"name": "PlaneFilterStage", "category": "calibration_filtering", "duration_ms": 5.1, "started_at": 10.0, "ended_at": 15.1},
                {"name": "BallClassificationStage", "category": "classification", "duration_ms": 3.4, "started_at": 15.2, "ended_at": 18.6},
            ]
        },
        input_modalities=["point_cloud"],
        objects=[{"object_id": 1}],
        rejected_objects=[],
        result_status="ok",
        result_error=None,
    )
    assert trace["pipeline_id"] == "3d_ball_inspection"
    statuses = {stage["stage_id"]: stage["status"] for stage in trace["stages"]}
    assert statuses["segmentation"] in {"success", "warning"}
    assert statuses["classification"] == "success"
    assert statuses["measurement"] in {"skipped", "warning"}


def test_pipeline_execution_reports_incompatible_stage() -> None:
    pipeline = {
        "id": "fusion",
        "stages": [
            {
                "id": "registration",
                "required_modalities": ["point_cloud", "rgb"],
                "implemented": True,
            }
        ],
    }
    trace = build_pipeline_execution_trace(
        pipeline=pipeline,
        artifacts=[],
        profiling=None,
        input_modalities=["point_cloud"],
        objects=[],
        rejected_objects=[],
        result_status="ok",
        result_error=None,
    )
    stage = trace["stages"][0]
    assert stage["status"] == "skipped"
    assert "missing modalities" in stage["warnings"][0]
