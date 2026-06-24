from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.ml.runtime import MLRuntimeResolver
from vision_3d_acquisition.ml.storage import MLStorage


def test_missing_deployment_resolves_with_fallback(tmp_path: Path) -> None:
    resolver = MLRuntimeResolver(tmp_path / "data")
    out = resolver.resolve_active_deployment(pipeline_id="p1", stage_id="classification", pipeline_family="25d")
    assert out["deployment"] is None
    assert out["diagnostics"]["fallback_mode"] == "heuristic"


def test_incompatible_deployment_feature_schema_and_pipeline_family(tmp_path: Path) -> None:
    storage = MLStorage(tmp_path / "data")
    storage.save(
        "models",
        "m1",
        {
            "id": "m1",
            "version": "1.0.0",
            "classifier_family": "random_forest",
            "promoted": True,
            "feature_schema": {"version": "9.9.9", "columns": ["wrong"]},
            "artifact_paths": {"model_path": ""},
            "backend": {"backend_name": "sklearn", "backend_version": "1.x", "feature_requirements": {}},
            "label_schema": {"classes": ["BALL_GOOD"]},
        },
    )
    storage.save(
        "deployments",
        "d1",
        {
            "id": "d1",
            "model_id": "m1",
            "target_pipeline_id": "p1",
            "target_stage_id": "classification",
            "pipeline_family": "wrong_family",
            "status": "active",
        },
    )
    resolver = MLRuntimeResolver(tmp_path / "data")
    out = resolver.resolve_active_deployment(pipeline_id="p1", stage_id="classification", pipeline_family="25d")
    assert out["diagnostics"]["fallback_mode"] == "heuristic"
    assert out["diagnostics"]["warnings"] or out["diagnostics"]["errors"]


def test_corrupted_model_artifact_fails_compatibility(tmp_path: Path) -> None:
    storage = MLStorage(tmp_path / "data")
    storage.save(
        "models",
        "m1",
        {
            "id": "m1",
            "version": "1.0.0",
            "classifier_family": "random_forest",
            "promoted": True,
            "feature_schema": {"version": "1.0.0", "columns": []},
            "artifact_paths": {"model_path": str((tmp_path / 'missing.pkl'))},
            "backend": {"backend_name": "sklearn", "backend_version": "1.x", "feature_requirements": {}},
            "label_schema": {"classes": ["BALL_GOOD"]},
        },
    )
    resolver = MLRuntimeResolver(tmp_path / "data")
    comp = resolver.validate_compatibility(deployment={"target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"}, model=storage.get("models", "m1") or {}, pipeline_id="p1", stage_id="classification", pipeline_family="25d")
    assert comp["compatible"] is False
    assert "model_file_missing" in comp["errors"]


def test_runtime_input_validation_required_metrics_and_calibration(tmp_path: Path) -> None:
    resolver = MLRuntimeResolver(tmp_path / "data")
    model = {
        "backend": {
            "feature_requirements": {
                "required_metric_fields": ["equivalent_diameter_mm", "sphere_fit_rmse_mm"],
                "requires_mm_calibration": True,
            }
        },
        "feature_schema": {"columns": ["feature_eccentricity"]},
    }
    out = resolver.validate_runtime_inputs(model=model, feature_row={"feature_eccentricity": 0.1}, object_metrics={}, calibration_available=False)
    assert out["compatible"] is False
    assert "mm_calibration_required" in out["errors"]
    assert "missing_required_metric_fields" in out["errors"]
