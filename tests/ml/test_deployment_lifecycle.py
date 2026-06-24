from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.ml.deployments import DeploymentService
from vision_3d_acquisition.ml.features.schemas import FEATURE_COLUMNS
from vision_3d_acquisition.ml.storage import MLStorage


def _seed_model(storage: MLStorage, model_id: str = "model_a", promoted: bool = True) -> None:
    model_file = storage.root / "models" / f"{model_id}.pkl"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"fake-model-binary")
    storage.save(
        "models",
        model_id,
        {
            "id": model_id,
            "version": "1.0.0",
            "promoted": promoted,
            "classifier_family": "random_forest",
            "feature_schema": {"version": "1.0.0", "columns": FEATURE_COLUMNS},
            "artifact_paths": {"model_path": str(model_file)},
            "backend": {"backend_name": "sklearn", "backend_version": "1.x", "feature_requirements": {}},
            "label_schema": {"classes": ["BALL_GOOD", "BALL_SCRAP"]},
        },
    )


def test_single_active_invariant_and_supersede(tmp_path: Path) -> None:
    storage = MLStorage(tmp_path / "data")
    _seed_model(storage, "model_a")
    _seed_model(storage, "model_b")
    svc = DeploymentService(tmp_path / "data")

    d1 = svc.create_deployment({"id": "dep1", "model_id": "model_a", "target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"})
    a1 = svc.activate(d1["id"])
    assert a1["status"] == "active"

    d2 = svc.create_deployment({"id": "dep2", "model_id": "model_b", "target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"})
    a2 = svc.activate(d2["id"])
    assert a2["status"] == "active"

    active = svc.list_active()
    same_target = [row for row in active if row["target_pipeline_id"] == "p1" and row["target_stage_id"] == "classification" and row["pipeline_family"] == "25d"]
    assert len(same_target) == 1
    assert str(same_target[0]["id"]).startswith("dep2")
    assert same_target[0]["status"] == "active"

    all_rows = svc.list_deployments()
    assert any(str(row.get("status")) == "superseded" and str(row.get("id", "")).startswith("dep1") for row in all_rows)
    assert any(str(row.get("id", "")).startswith("dep1") for row in all_rows)
    assert any(str(row.get("id", "")).startswith("dep2") for row in all_rows)


def test_rollback_lineage_and_immutable_snapshots(tmp_path: Path) -> None:
    storage = MLStorage(tmp_path / "data")
    _seed_model(storage, "model_a")
    svc = DeploymentService(tmp_path / "data")

    d1 = svc.create_deployment({"id": "dep1", "model_id": "model_a", "target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"})
    a1 = svc.activate(d1["id"])
    rolled = svc.rollback(a1["id"])

    assert str(rolled.get("rollback_from") or "") != ""
    rows = svc.list_deployments()
    assert any(str(row.get("status")) == "rolled_back" for row in rows)

    first = next(row for row in rows if str(row.get("id", "")).startswith("dep1"))
    compat_before = first.get("compatibility_snapshot")
    # ensure snapshot exists and persists in historical records
    assert isinstance(compat_before, dict)


def test_activation_and_deactivation_transitions(tmp_path: Path) -> None:
    storage = MLStorage(tmp_path / "data")
    _seed_model(storage, "model_a")
    svc = DeploymentService(tmp_path / "data")

    d1 = svc.create_deployment({"id": "dep1", "model_id": "model_a", "target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"})
    assert d1["status"] == "inactive"
    a1 = svc.activate(d1["id"])
    assert a1["status"] == "active"
    de = svc.deactivate(a1["id"])
    assert de["status"] == "inactive"
    assert de.get("deactivated_at") is not None


def test_invalid_activation_does_not_supersede_existing_active(tmp_path: Path) -> None:
    storage = MLStorage(tmp_path / "data")
    _seed_model(storage, "model_good")
    _seed_model(storage, "model_bad")
    # corrupt bad model compatibility deterministically
    bad = storage.get("models", "model_bad") or {}
    bad["feature_schema"] = {"version": "9.9.9", "columns": ["wrong_feature"]}
    storage.save("models", "model_bad", bad)
    svc = DeploymentService(tmp_path / "data")

    good_dep = svc.create_deployment(
        {"id": "dep_good", "model_id": "model_good", "target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"}
    )
    good_active = svc.activate(good_dep["id"])
    assert good_active["status"] == "active"

    bad_dep = svc.create_deployment(
        {"id": "dep_bad", "model_id": "model_bad", "target_pipeline_id": "p1", "target_stage_id": "classification", "pipeline_family": "25d"}
    )
    bad_activation = svc.activate(bad_dep["id"])
    assert bad_activation["status"] == "invalid"

    active = svc.list_active()
    same_target = [row for row in active if row["target_pipeline_id"] == "p1" and row["target_stage_id"] == "classification" and row["pipeline_family"] == "25d"]
    assert len(same_target) == 1
    assert str(same_target[0]["id"]).startswith("dep_good")
