from __future__ import annotations

import json
from pathlib import Path

from scripts import sensor_studio_cli as cli
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


def test_cli_ml_create_set(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "create-set",
            "--data-dir",
            str(data_dir),
            "--dataset-id",
            "d1",
            "--ml-set-id",
            "balls_scrap_classifier_v1",
            "--name",
            "Balls vs Scrap Classifier v1",
            "--task-type",
            "classification",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Created ML set" in out
    assert service.get_ml_set("d1", "balls_scrap_classifier_v1") is not None


def test_cli_ml_add_takes_from_file(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    _write_take(data_dir, "t2")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.create_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", name="Balls", task_type="classification")
    take_file = tmp_path / "takes.txt"
    take_file.write_text("t1\nt2\nt2\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "add-takes",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "balls_scrap_classifier_v1",
            "--take-file",
            str(take_file),
            "--split",
            "unassigned",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "ML set membership update" in out
    memberships = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1")
    assert len(memberships) == 2


def test_cli_ml_list_set(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.create_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", name="Balls", task_type="classification")
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", take_id="t1", split="unassigned")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "list-set",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "balls_scrap_classifier_v1",
            "--show-memberships",
            "--limit",
            "5",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "ML Set" in out
    assert "Split distribution" in out
    assert "Resolved MLSet" in out
    assert "unassigned: 1" in out


def test_cli_ml_list_set_ambiguous_without_dataset_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")
    service.create_dataset(dataset_id="dataset_b", name="B")
    service.create_ml_set(dataset_id="dataset_a", ml_set_id="ml_set_x", name="XA", task_type="classification")
    service.create_ml_set(dataset_id="dataset_b", ml_set_id="ml_set_x", name="XB", task_type="classification")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "list-set",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "ml_set_x",
        ],
    )
    code = cli.main()
    err = capsys.readouterr().err
    assert code == 2
    assert "Multiple MLSets named 'ml_set_x' exist." in err
    assert "--dataset-id" in err


def test_cli_ml_list_set_explicit_dataset_resolves(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")
    service.create_dataset(dataset_id="dataset_b", name="B")
    service.create_ml_set(dataset_id="dataset_a", ml_set_id="ml_set_x", name="XA", task_type="classification")
    service.create_ml_set(dataset_id="dataset_b", ml_set_id="ml_set_x", name="XB", task_type="classification")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "list-set",
            "--data-dir",
            str(data_dir),
            "--dataset-id",
            "dataset_b",
            "--ml-set-id",
            "ml_set_x",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "dataset_id: dataset_b" in out


def test_cli_ml_list_set_missing_fails_cleanly(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="dataset_a", name="A")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "list-set",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "missing_set",
        ],
    )
    code = cli.main()
    err = capsys.readouterr().err
    assert code == 2
    assert "MLSet 'missing_set' not found." in err


def test_cli_ml_assign_splits_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", name="Balls", task_type="classification")
    for take_id, obj_id in [("t1", "obj_a"), ("t2", "obj_a"), ("t3", "obj_b"), ("t4", "obj_c")]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", take_id=take_id, physical_object_id=obj_id)

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "assign-splits",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "balls_scrap_classifier_v1",
            "--by-physical-object-id",
            "--train",
            "0.7",
            "--validation",
            "0.15",
            "--test",
            "0.15",
            "--seed",
            "42",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "[DRY RUN]" in out
    assert "Projected split distribution:" in out
    assert "Grouped objects:" in out


def test_cli_ml_assign_splits_apply(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", name="Balls", task_type="classification")
    for take_id, obj_id in [("t1", "obj_a"), ("t2", "obj_a"), ("t3", "obj_b"), ("t4", "obj_c")]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1", take_id=take_id, physical_object_id=obj_id)

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "assign-splits",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "balls_scrap_classifier_v1",
            "--seed",
            "42",
            "--apply",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "[APPLY]" in out
    assert "Wrote operation log:" in out
    logs = list((data_dir / "runtime" / "logs").glob("ml_assign_splits_*.json"))
    assert logs
    memberships = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="balls_scrap_classifier_v1")
    assert all(str(row.get("split") or "") != "unassigned" for row in memberships if bool(row.get("include", True)))


def test_cli_ml_reprocess_set_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_reprocess", name="Reprocess", task_type="classification")
    for take_id, split in [("t1", "train"), ("t2", "train"), ("t3", "test")]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_reprocess", take_id=take_id, split=split, physical_object_id=f"obj_{take_id}")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "reprocess-set",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "set_reprocess",
            "--pipeline-id",
            "mining_steel_ball_classification_25d",
            "--split",
            "train",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "[DRY RUN]" in out
    assert "Planned processing:" in out


def test_cli_ml_reprocess_set_apply_logs(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_reprocess", name="Reprocess", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="set_reprocess", take_id="t1", split="train", physical_object_id="obj_t1")

    def _fake_dispatch(**kwargs):
        return {"status": "ok", "take_id": kwargs.get("take_id")}

    monkeypatch.setattr("vision_3d_acquisition.api.processing_dispatch.dispatch_take_processing", _fake_dispatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "reprocess-set",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "set_reprocess",
            "--pipeline-id",
            "mining_steel_ball_classification_25d",
            "--classifier-rules",
            "configs/classifiers/demo_rules.json",
            "--apply",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "[APPLY]" in out
    assert "classifier_rules: configs/classifiers/demo_rules.json" in out
    logs = list((data_dir / "runtime" / "logs").glob("ml_reprocess_set_*.json"))
    assert logs


def test_cli_ml_export_features(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_export", name="Export", task_type="classification")
    _write_take(data_dir, "t1")
    service.upsert_take_metadata(
        take_id="t1",
        dataset_id="d1",
        session_id="s1",
        updates={"expected_class": "ball"},
        source_metadata={"session_id": "session_live"},
    )
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
                "summary": {"object_count": 1, "confidence": 0.9},
                "objects": [{"object_id": 1, "diameter_mm": 84.2}],
            }
        ),
        encoding="utf-8",
    )
    (processed / "feature_vector.json").write_text(json.dumps({"features": {"valid_pixel_ratio": 0.94}}), encoding="utf-8")
    (processed / "quality_flags.json").write_text(json.dumps({"flags": [{"code": "BORDER_TOUCH"}]}), encoding="utf-8")
    (processed / "feature_provenance.json").write_text(
        json.dumps({"equivalent_diameter_mm": {"source_stage": "geometry", "validity": "warning"}}),
        encoding="utf-8",
    )
    output = data_dir / "ml_exports" / "features.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "export-features",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "set_export",
            "--pipeline-id",
            "mining_steel_ball_classification_25d",
            "--output",
            str(output),
            "--include-diagnostics",
            "--include-invalidity-flags",
            "--include-provenance-summary",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Exported ML features" in out
    assert "include_diagnostics: yes" in out
    assert output.is_file()


def test_cli_ml_import_manifest_dry_run_and_apply(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="set_manifest", name="Manifest", task_type="classification")
    for take_id in ["t1", "t2"]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    manifest = data_dir / "labels" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("physical_object_id,label,take_ids\nobj_1,ahuevada,t1;t2\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "import-manifest",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "set_manifest",
            "--manifest",
            str(manifest),
        ],
    )
    dry_code = cli.main()
    dry_out = capsys.readouterr().out
    assert dry_code == 0
    assert "[DRY RUN]" in dry_out
    assert service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_manifest") == []

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "import-manifest",
            "--data-dir",
            str(data_dir),
            "--ml-set-id",
            "set_manifest",
            "--manifest",
            str(manifest),
            "--apply",
        ],
    )
    apply_code = cli.main()
    apply_out = capsys.readouterr().out
    assert apply_code == 0
    assert "[APPLY]" in apply_out
    memberships = service.list_ml_set_memberships(dataset_id="d1", ml_set_id="set_manifest")
    assert len(memberships) == 2
    logs = list((data_dir / "runtime" / "logs").glob("ml_import_manifest_*.json"))
    assert logs


def test_cli_ml_list_rule_sets_text_output(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_dir = tmp_path / "configs" / "classifiers"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "mining_ball_rules_tuned_20260529.json").write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "tuned_20260529T000000Z",
                "description": "Tuned using balls_scrap_classifier_v1",
                "metadata": {
                    "dataset_id": "bolas-2-5-1",
                    "ml_set_id": "balls_scrap_classifier_v1",
                    "optimized_metric": "macro_f1",
                    "validation_macro_f1": 0.91,
                },
                "params": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["sensor_studio_cli.py", "ml", "list-rule-sets", "--data-dir", str(tmp_path / "data")],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Available classifier rule sets" in out
    assert "id: mining_ball_rules_tuned_20260529" in out
    assert "classifier_id: mining_steel_ball_classification_25d_rules" in out
    assert "optimized_metric: macro_f1" in out


def test_cli_ml_list_rule_sets_json_output_and_filter(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_dir = tmp_path / "configs" / "classifiers"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "rules_a.json").write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "v1",
                "description": "A",
                "params": {},
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "rules_b.json").write_text(
        json.dumps(
            {
                "classifier_id": "other_classifier_rules",
                "version": "v2",
                "description": "B",
                "params": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "list-rule-sets",
            "--data-dir",
            str(tmp_path / "data"),
            "--json",
            "--classifier-id",
            "other_classifier_rules",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["rule_set_id"] == "rules_b"


def test_cli_ml_show_active_rule_set_json_runtime_override(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_dir = tmp_path / "configs" / "classifiers"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / "rules_runtime.json"
    cfg.write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "runtime_v1",
                "metadata": {"optimized_metric": "macro_f1"},
                "params": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "ml",
            "show-active-rule-set",
            "--pipeline-id",
            "mining_steel_ball_classification_25d",
            "--classifier-rules",
            str(cfg),
            "--json",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["rule_set_source"] == "runtime_override"
    assert payload["rule_set_version"] == "runtime_v1"
