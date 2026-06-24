from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest

from vision_3d_acquisition.datasets import DatasetService


def _write_take(data_dir: Path, take_id: str, *, session_id: str = "session_live") -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-25T10:00:00Z", "session_id": session_id, "files": {"rgb": "rgb.png"}}),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def _setup_ml_set_with_objects(data_dir: Path, ml_set_id: str = "set_split") -> DatasetService:
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id=ml_set_id, name="Split Set", task_type="classification")
    for take_id, obj_id in [
        ("t1", "obj_a"),
        ("t2", "obj_a"),
        ("t3", "obj_b"),
        ("t4", "obj_c"),
        ("t5", "obj_d"),
        ("t6", "obj_e"),
    ]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id=ml_set_id, take_id=take_id, split="unassigned", physical_object_id=obj_id)
    return service


def test_create_ml_set_and_duplicate_and_overwrite(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")

    created = service.create_ml_set(
        dataset_id="d1",
        ml_set_id="set_a",
        name="Set A",
        task_type="classification",
    )
    assert created["id"] == "set_a"
    assert created["dataset_id"] == "d1"
    assert service.get_ml_set("d1", "set_a") is not None

    with pytest.raises(ValueError):
        service.create_ml_set(dataset_id="d1", ml_set_id="set_a", name="Set A", task_type="classification")

    overwritten = service.create_ml_set(
        dataset_id="d1",
        ml_set_id="set_a",
        name="Set A v2",
        task_type="classification",
        overwrite=True,
    )
    assert overwritten["name"] == "Set A v2"


def test_membership_add_update_and_multi_ml_set_membership(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.create_ml_set(dataset_id="d1", ml_set_id="set_a", name="Set A", task_type="classification")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_b", name="Set B", task_type="benchmark")

    add_result = service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_a", take_id="t1", split="train")
    assert add_result["status"] == "added"
    memberships = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_a")
    assert len(memberships) == 1
    assert memberships[0]["split"] == "train"

    update_result = service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_a", take_id="t1", split="validation", notes="reviewed")
    assert update_result["status"] == "updated"
    memberships_after = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_a")
    assert len(memberships_after) == 1
    assert memberships_after[0]["split"] == "validation"
    assert memberships_after[0]["notes"] == "reviewed"

    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_b", take_id="t1", split="holdout")
    memberships_b = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_b")
    assert len(memberships_b) == 1
    assert memberships_b[0]["take_id"] == "t1"


def test_membership_dataset_mismatch_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_dataset(dataset_id="d2", name="Dataset 2")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.create_ml_set(dataset_id="d2", ml_set_id="set_wrong", name="Wrong", task_type="classification")

    with pytest.raises(ValueError):
        service.add_take_to_ml_set(dataset_id="d2", ml_set_id="set_wrong", take_id="t1", split="train")


def test_list_ml_set_distribution_summary_values(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    _write_take(data_dir, "t2")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.create_ml_set(dataset_id="d1", ml_set_id="set_dist", name="Dist", task_type="classification")
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_dist", take_id="t1", split="train", include=True)
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_dist", take_id="t2", split="test", include=False)

    memberships = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_dist")
    assert len(memberships) == 2
    split_counts = {}
    for row in memberships:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    assert split_counts == {"train": 1, "test": 1}


def test_resolve_ml_set_unique_auto_resolves(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")
    service.create_ml_set(dataset_id="dataset_a", ml_set_id="ml_set_x", name="X", task_type="classification")

    resolved = service.resolve_ml_set(ml_set_id="ml_set_x")
    assert resolved["dataset_id"] == "dataset_a"
    assert resolved["id"] == "ml_set_x"


def test_resolve_ml_set_ambiguous_requires_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")
    service.create_dataset(dataset_id="dataset_b", name="B")
    service.create_ml_set(dataset_id="dataset_a", ml_set_id="ml_set_x", name="X A", task_type="classification")
    service.create_ml_set(dataset_id="dataset_b", ml_set_id="ml_set_x", name="X B", task_type="classification")

    with pytest.raises(ValueError, match="Multiple MLSets named 'ml_set_x' exist."):
        service.resolve_ml_set(ml_set_id="ml_set_x")


def test_resolve_ml_set_with_explicit_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")
    service.create_dataset(dataset_id="dataset_b", name="B")
    service.create_ml_set(dataset_id="dataset_a", ml_set_id="ml_set_x", name="X A", task_type="classification")
    service.create_ml_set(dataset_id="dataset_b", ml_set_id="ml_set_x", name="X B", task_type="classification")

    resolved = service.resolve_ml_set(ml_set_id="ml_set_x", dataset_id="dataset_b")
    assert resolved["dataset_id"] == "dataset_b"
    assert resolved["name"] == "X B"


def test_resolve_ml_set_missing_fails_cleanly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")

    with pytest.raises(ValueError, match="MLSet 'missing_set' not found."):
        service.resolve_ml_set(ml_set_id="missing_set")


def test_assign_splits_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir)

    a = service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=42, dry_run=True)
    b = service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=42, dry_run=True)
    assert a["assignments"] == b["assignments"]


def test_assign_splits_changes_with_different_seed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir)

    a = service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=42, dry_run=True)
    b = service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=7, dry_run=True)
    assert a["assignments"] != b["assignments"]


def test_assign_splits_prevents_object_leakage(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir)

    result = service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=42, dry_run=True)
    split_by_take = {}
    for row in result["assignments"]:
        for take_id in row["take_ids"]:
            split_by_take[take_id] = row["split"]
    assert split_by_take["t1"] == split_by_take["t2"]


def test_assign_splits_missing_physical_object_id_fails(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_split", name="Split Set", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_split", take_id="t1", split="unassigned", physical_object_id=None)

    with pytest.raises(ValueError, match="missing physical_object_id for takes: t1"):
        service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", by_physical_object_id=True, dry_run=True)


def test_assign_splits_ratio_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", train_ratio=0.5, validation_ratio=0.1, test_ratio=0.1, dry_run=True)


def test_assign_splits_dry_run_does_not_write(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir)
    before = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_split")
    before_splits = {row["take_id"]: row["split"] for row in before}

    service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=42, dry_run=True)
    after = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_split")
    after_splits = {row["take_id"]: row["split"] for row in after}
    assert before_splits == after_splits


def test_assign_splits_apply_updates_memberships(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir)
    service.assign_ml_set_splits(ml_set_id="set_split", dataset_id="d1", seed=42, dry_run=False)

    after = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_split")
    splits = {str(row["split"]) for row in after}
    assert "unassigned" not in splits


def test_reprocess_ml_set_dry_run_no_processing(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = _setup_ml_set_with_objects(data_dir, ml_set_id="set_reprocess")
    called: list[str] = []

    def _fake_dispatch(**kwargs):
        called.append(str(kwargs.get("take_id")))
        return {"status": "ok"}

    monkeypatch.setattr("vision_3d_acquisition.api.processing_dispatch.dispatch_take_processing", _fake_dispatch)
    result = service.reprocess_ml_set(
        ml_set_id="set_reprocess",
        dataset_id="d1",
        pipeline_id="mining_steel_ball_classification_25d",
        splits=["train"],
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert called == []


def test_reprocess_ml_set_apply_with_split_filter_and_failure_isolation(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_reprocess", name="Reprocess Set", task_type="classification")
    for take_id, split in [("t1", "train"), ("t2", "train"), ("t3", "test")]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_reprocess", take_id=take_id, split=split, physical_object_id=f"obj_{take_id}")

    called: list[dict[str, object]] = []

    def _fake_dispatch(**kwargs):
        take = str(kwargs.get("take_id"))
        called.append(kwargs)
        if take == "t2":
            raise RuntimeError("boom")
        return {"status": "ok"}

    monkeypatch.setattr("vision_3d_acquisition.api.processing_dispatch.dispatch_take_processing", _fake_dispatch)
    result = service.reprocess_ml_set(
        ml_set_id="set_reprocess",
        dataset_id="d1",
        pipeline_id="mining_steel_ball_classification_25d",
        splits=["train"],
        dry_run=False,
        classifier_rules_path="configs/classifiers/mining_ball_rules_tuned_20260529.json",
    )
    assert sorted(str(item.get("take_id")) for item in called) == ["t1", "t2"]
    stage_params = [item.get("stage_params") for item in called]
    assert all(isinstance(item, dict) for item in stage_params)
    assert all(((item or {}).get("classify_25d") or {}).get("classifier_rules_path") == "configs/classifiers/mining_ball_rules_tuned_20260529.json" for item in stage_params)
    assert result["success_count"] == 1
    assert result["failure_count"] == 1


def test_export_ml_set_features_csv_and_deterministic_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_export", name="Export Set", task_type="classification")
    for take_id in ["t1", "t2"]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(
            take_id=take_id,
            dataset_id="d1",
            session_id="s1",
            updates={"expected_class": "ball"},
            source_metadata={"session_id": "session_live"},
        )
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_export", take_id=take_id, split="train", physical_object_id=f"obj_{take_id}")
        processed = data_dir / "processed" / take_id
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "result.json").write_text(
            json.dumps(
                {
                    "take_id": take_id,
                    "processed_at": "2026-05-25T10:00:00Z",
                    "run_id": f"run_{take_id}",
                    "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "pipeline_family": "25d"},
                    "classification": {
                        "classifier_engine": "mining_steel_ball_classification_25d",
                        "rule_set_id": "builtin_default",
                        "rule_set_version": "builtin",
                        "rule_set_source": "builtin_default",
                    },
                    "summary": {"object_count": 1, "confidence": 0.9},
                    "timing_ms": {"total": 12.3},
                    "objects": [{"object_id": 1, "diameter_mm": 84.2, "circularity": 0.97}],
                }
            ),
            encoding="utf-8",
        )

    output = data_dir / "ml_exports" / "features.csv"
    result = service.export_ml_set_features(
        ml_set_id="set_export",
        dataset_id="d1",
        pipeline_id="mining_steel_ball_classification_25d",
        output_path=output,
    )
    assert output.is_file()
    assert result["row_count"] == 2
    with output.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert reader.fieldnames == result["columns"]
    assert rows[0]["take_id"] == "t1"
    assert "summary_object_count" in result["columns"]
    assert "classifier_engine" in result["columns"]
    assert "rule_set_id" in result["columns"]
    assert "rule_set_version" in result["columns"]
    assert "rule_set_source" in result["columns"]


def test_export_ml_set_features_require_processed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_export", name="Export Set", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_export", take_id="t1", split="train", physical_object_id="obj_t1")

    with pytest.raises(ValueError, match="missing processing output"):
        service.export_ml_set_features(
            ml_set_id="set_export",
            dataset_id="d1",
            pipeline_id="mining_steel_ball_classification_25d",
            output_path=data_dir / "ml_exports" / "features.csv",
            require_processed=True,
        )


def test_export_ml_set_features_with_diagnostics_flags_and_provenance(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_export", name="Export Set", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_export", take_id="t1", split="train", physical_object_id="obj_t1")
    processed = data_dir / "processed" / "t1"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "result.json").write_text(
        json.dumps(
            {
                "take_id": "t1",
                "processed_at": "2026-05-25T10:00:00Z",
                "run_id": "run_t1",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "pipeline_family": "25d"},
                "summary": {"object_count": 1},
                "objects": [{"object_id": 1, "diameter_mm": 84.2}],
            }
        ),
        encoding="utf-8",
    )
    (processed / "feature_vector.json").write_text(json.dumps({"features": {"valid_pixel_ratio": 0.95}}), encoding="utf-8")
    (processed / "quality_flags.json").write_text(json.dumps({"flags": [{"code": "BORDER_TOUCH"}]}), encoding="utf-8")
    (processed / "feature_provenance.json").write_text(
        json.dumps({"equivalent_diameter_mm": {"source_stage": "geometry", "validity": "warning"}}),
        encoding="utf-8",
    )

    output = data_dir / "ml_exports" / "features.csv"
    result = service.export_ml_set_features(
        ml_set_id="set_export",
        dataset_id="d1",
        pipeline_id="mining_steel_ball_classification_25d",
        output_path=output,
        include_diagnostics=True,
        include_invalidity_flags=True,
        include_provenance_summary=True,
    )
    assert "diag_valid_pixel_ratio" in result["columns"]
    assert "diagnostic_flag_count" in result["columns"]
    assert "provenance_source_stage_count" in result["columns"]


def test_import_ml_manifest_csv_and_idempotent_update(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_manifest", name="Manifest", task_type="classification")
    for take_id in ["t1", "t2", "t3"]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    manifest = data_dir / "labels" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "physical_object_id,label,d1_mm,take_ids\n"
        "obj_1,ahuevada,91.5,t1;t2\n"
        "obj_2,cubo,80,t3\n",
        encoding="utf-8",
    )

    dry = service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=manifest, dry_run=True)
    assert dry["membership_added"] == 3
    assert service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_manifest") == []

    applied = service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=manifest, dry_run=False)
    assert applied["membership_added"] == 3
    memberships = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_manifest")
    assert len(memberships) == 3
    assert memberships[0]["expected_label"] in {"ahuevada", "cubo"}

    # Idempotent reimport should update, not duplicate.
    again = service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=manifest, dry_run=False)
    assert again["membership_updated"] == 3
    assert len(service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_manifest")) == 3


def test_import_ml_manifest_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_manifest", name="Manifest", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    manifest = data_dir / "labels" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps([{"physical_object_id": "obj_1", "label": "mitad", "take_ids": ["t1"], "split": "train"}]),
        encoding="utf-8",
    )
    applied = service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=manifest, dry_run=False)
    assert applied["membership_added"] == 1


def test_import_ml_manifest_conflicting_object_assignment_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_manifest", name="Manifest", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    manifest = data_dir / "labels" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "physical_object_id,take_ids\n"
        "obj_1,t1\n"
        "obj_2,t1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting physical_object_id assignments"):
        service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=manifest, dry_run=True)


def test_import_ml_manifest_dataset_mismatch_and_malformed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_dataset(dataset_id="d2", name="Dataset 2")
    service.create_session(dataset_id="d2", session_id="s2", name="Session 2")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_manifest", name="Manifest", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d2", session_id="s2", updates={}, source_metadata={"session_id": "session_live"})
    manifest = data_dir / "labels" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("physical_object_id,take_ids\nobj_1,t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset mismatch"):
        service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=manifest, dry_run=True)

    bad = data_dir / "labels" / "bad.csv"
    bad.write_text("physical_object_id,take_ids\n, \n", encoding="utf-8")
    with pytest.raises(ValueError):
        service.import_ml_manifest(ml_set_id="set_manifest", dataset_id="d1", manifest_path=bad, dry_run=True)
