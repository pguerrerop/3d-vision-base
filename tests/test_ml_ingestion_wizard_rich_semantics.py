from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.api.main import dataset_ml_sets
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml.ingestion_wizard import IngestionWizardService


RICH_TSV = "\n".join(
    [
        "object_id\tsource_row\traw_label\tlabel\tnormalized_class\tsuperclass\tlabel_policy\treview_required\timage_ref\td1_mm\td2_mm\td3_mm\tnote",
        "obj_0002\t1\tAhuevada\tdeformed\tahuevada\tBALL_SCRAP\tinclude\tfalse\t2026-05-25T150423_025\t91.5\t90.6\t72.4\tkeep",
        "obj_0002\t1\tAhuevada\tdeformed\tahuevada\tBALL_SCRAP\tinclude\tfalse\t2026-05-25T150423_026\t91.5\t90.6\t72.4\tkeep",
        "obj_0019\t19\tAhuevada?\tdeformed\tahuevada\tBALL_SCRAP\treview\ttrue\t2026-05-29T221714_022\t86\t86\t74\treview",
    ]
) + "\n"


def _settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def _write_take(settings: ApiSettings, take_id: str, session_id: str) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "metadata.json").write_text(json.dumps({"take_id": take_id, "created_at": "2026-06-03T12:00:00Z", "session_id": session_id}), encoding="utf-8")
    (take_dir / "READY").touch()


def _setup_run(tmp_path: Path, *, take_ids: list[str] | None = None) -> tuple[ApiSettings, IngestionWizardService, dict[str, str]]:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    for take_id in take_ids or ["2026-05-25T150423_025", "2026-05-25T150423_026", "2026-05-29T221714_022"]:
        _write_take(settings, take_id, "runtime")
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "runtime"})
    wizard = IngestionWizardService(settings.data_dir)
    run = wizard.create_run(dataset_id="d1", session_ids=["s1"], name="rich")
    return settings, wizard, run


def _build_policy_rich_tsv(*, default_trainable_count: int = 156, review_count: int = 67, default_trainable_objects: int = 29, review_objects: int = 13) -> tuple[str, list[str]]:
    rows = ["object_id\tsource_row\traw_label\tlabel\tnormalized_class\tsuperclass\tlabel_policy\treview_required\timage_ref\td1_mm\td2_mm\td3_mm\tnote"]
    take_ids: list[str] = []
    total = default_trainable_count + review_count
    trainable_object_ids = [f"obj_{index + 1:04d}" for index in range(default_trainable_objects)]
    review_object_ids = [f"obj_{default_trainable_objects + index + 1:04d}" for index in range(review_objects)]
    for index in range(total):
        take_id = f"2026-05-25T150423_{index + 1:03d}"
        take_ids.append(take_id)
        if index < default_trainable_count:
            object_id = trainable_object_ids[index % len(trainable_object_ids)]
            rows.append(
                f"{object_id}\t{index + 1}\tBuena\tgood\tbuena\tBALL_GOOD\tinclude\tfalse\t{take_id}\t90\t90\t70\ttrainable"
            )
        else:
            object_id = review_object_ids[(index - default_trainable_count) % len(review_object_ids)]
            rows.append(
                f"{object_id}\t{index + 1}\tAhuevada?\tdeformed\tahuevada\tBALL_SCRAP\treview\ttrue\t{take_id}\t88\t88\t72\treview"
            )
    return "\n".join(rows) + "\n", take_ids


def test_rich_tsv_parse_preserves_semantic_columns(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    payload = wizard.ingest_table(run["run_id"], content=RICH_TSV, input_format="tsv")
    row = payload["rows"][0]
    assert payload["schema"]["object_id_column"] == "object_id"
    assert row["physical_object_id"] == "obj_0002"
    assert row["raw_label"] == "Ahuevada"
    assert row["normalized_class"] == "ahuevada"
    assert row["superclass"] == "BALL_SCRAP"
    assert row["label_policy"] == "include"
    assert row["review_required"] is False
    assert row["extra_fields"] == {"note": "keep"}


def test_object_id_grouping_and_review_policy_handling(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    wizard.ingest_table(run["run_id"], content=RICH_TSV, input_format="tsv")
    reconciled = wizard.reconcile(run["run_id"])
    rows = reconciled["rows"]
    assert rows[0]["physical_object_id"] == "obj_0002"
    assert rows[1]["physical_object_id"] == "obj_0002"
    assert reconciled["diagnostics"]["grouping_summary"]["physical_object_count"] == 2
    assert reconciled["diagnostics"]["grouping_summary"]["rows_missing_object_id"] == 0
    review_row = next(row for row in rows if row["physical_object_id"] == "obj_0019")
    assert review_row["review_required"] is True
    assert review_row["label_policy"] == "review"
    assert review_row["include"] is True
    assert review_row["normalization_warnings"] == []


def test_valid_table_provided_canonical_values_produce_no_warning(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    wizard.ingest_table(run["run_id"], content=RICH_TSV, input_format="tsv")
    reconciled = wizard.reconcile(run["run_id"])
    assert all(not row["normalization_warnings"] for row in reconciled["rows"])
    preview_rows = reconciled["diagnostics"]["normalization_preview"]
    assert all(not item["warnings"] for item in preview_rows)


def test_invalid_superclass_warns_and_blocks_manifest(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    invalid = RICH_TSV.replace("BALL_SCRAP", "NOT_A_CLASS", 1)
    wizard.ingest_table(run["run_id"], content=invalid, input_format="tsv")
    reconciled = wizard.reconcile(run["run_id"])
    assert any("Unknown superclass: NOT_A_CLASS" in warning for row in reconciled["rows"] for warning in row["normalization_warnings"])
    assert any("invalid_superclass" in row["normalization_warning_codes"] for row in reconciled["rows"])
    with pytest.raises(ValueError, match="invalid normalized_class/superclass values"):
        wizard.generate_manifest(run["run_id"])


def test_unknown_raw_label_with_valid_table_canonical_is_accepted(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    content = RICH_TSV.replace("Ahuevada\tdeformed\tahuevada\tBALL_SCRAP", "Misterio\tdeformed\tahuevada\tBALL_SCRAP", 1)
    wizard.ingest_table(run["run_id"], content=content, input_format="tsv")
    reconciled = wizard.reconcile(run["run_id"])
    first = reconciled["rows"][0]
    assert first["raw_label"] == "Misterio"
    assert first["normalized_class"] == "ahuevada"
    assert first["superclass"] == "BALL_SCRAP"
    assert first["normalization_warnings"] == []


def test_legacy_table_without_canonical_values_warns_on_unknown_label(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    content = "label\timage_ref\nmisterio\t2026-05-25T150423_025\n"
    wizard.ingest_table(run["run_id"], content=content, input_format="tsv")
    reconciled = wizard.reconcile(run["run_id"])
    assert any("Unmapped raw label: misterio" in warning for warning in reconciled["rows"][0]["normalization_warnings"])


def test_rich_tsv_policy_counts_keep_review_rows_out_of_default_training(tmp_path: Path) -> None:
    content, take_ids = _build_policy_rich_tsv()
    settings, wizard, run = _setup_run(tmp_path, take_ids=take_ids)
    wizard.ingest_table(run["run_id"], content=content, input_format="tsv")
    wizard.reconcile(run["run_id"])
    manifest = wizard.generate_manifest(run["run_id"])
    materialized = wizard.materialize(run["run_id"], "mls_policy")

    manifest_csv = Path(manifest["files"]["label_manifest_csv"])
    manifest_rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
    assert len(manifest_rows) == 223
    assert sum(row["include"] == "True" for row in manifest_rows) == 223
    assert sum(row["default_trainable"] == "True" for row in manifest_rows) == 156
    assert sum(row["review_required"] == "True" for row in manifest_rows) == 67
    assert sum(row["label_policy"] in {"exclude", "reference"} for row in manifest_rows) == 0

    memberships = DatasetService(settings.data_dir).list_ml_set_memberships(dataset_id="d1", ml_set_id="mls_policy")
    assert len(memberships) == 223
    assert sum(bool(row.get("include", True)) for row in memberships) == 223
    assert sum(bool(row.get("default_trainable", False)) for row in memberships) == 156
    assert sum(bool(row.get("trainable", False)) for row in memberships) == 156
    assert sum(bool(str(row.get("physical_object_id") or "").strip()) for row in memberships) == 223
    assert len({str(row.get("physical_object_id") or "") for row in memberships if str(row.get("physical_object_id") or "")}) == 42
    persisted_ml_set = json.loads((settings.data_dir / "datasets" / "dataset_d1" / "ml_sets" / "ml_set_mls_policy" / "ml_set.json").read_text(encoding="utf-8"))
    assert persisted_ml_set["membership"]["mode"] == "physical_object_id"
    assert persisted_ml_set["membership_mode"] == "physical_object_id"
    assert persisted_ml_set["membership_count"] == 223
    assert persisted_ml_set["physical_object_count"] == 42
    assert persisted_ml_set["default_trainable_rows"] == 156
    assert persisted_ml_set["default_trainable_objects"] == 29
    assert persisted_ml_set["review_required_rows"] == 67
    assert persisted_ml_set["review_required_objects"] == 13
    assert persisted_ml_set["split_status"] == "unassigned"
    assert materialized["ok"] is True
    assert materialized["dataset_id"] == "d1"
    assert materialized["ml_set_id"] == "mls_policy"
    assert materialized["membership_count"] == 223
    assert materialized["physical_object_count"] == 42
    assert materialized["default_trainable_rows"] == 156
    assert materialized["default_trainable_objects"] == 29
    assert materialized["review_required_rows"] == 67
    assert materialized["review_required_objects"] == 13
    assert materialized["split_status"] == "unassigned"
    assert materialized["errors"] == []
    assert materialized["validation_report"]["kept_in_ml_set_count"] == 223
    assert materialized["validation_report"]["default_trainable_count"] == 156
    assert materialized["validation_report"]["trainable_count"] == 156
    listed = dataset_ml_sets("d1", settings)
    listed_row = next(item for item in listed if item["id"] == "mls_policy")
    assert listed_row["membership_mode"] == "physical_object_id"
    assert listed_row["member_count"] == 223
    assert listed_row["physical_object_count"] == 42
    assert listed_row["default_trainable_count"] == 156
    assert listed_row["review_required_count"] == 67


def test_conflicting_labels_within_object_id_block_materialization(tmp_path: Path) -> None:
    _settings_value, wizard, run = _setup_run(tmp_path)
    conflicting = RICH_TSV.replace("BALL_SCRAP\tinclude\tfalse\t2026-05-25T150423_026", "BALL_GOOD\tinclude\tfalse\t2026-05-25T150423_026")
    wizard.ingest_table(run["run_id"], content=conflicting, input_format="tsv")
    reconciled = wizard.reconcile(run["run_id"])
    assert reconciled["diagnostics"]["object_conflicts"]
    with pytest.raises(ValueError, match="physical object conflicts remain"):
        wizard.materialize(run["run_id"], "mls_rich")


def test_manifest_and_materialized_memberships_preserve_rich_fields(tmp_path: Path) -> None:
    settings, wizard, run = _setup_run(tmp_path)
    wizard.ingest_table(run["run_id"], content=RICH_TSV, input_format="tsv")
    wizard.reconcile(run["run_id"])
    manifest = wizard.generate_manifest(run["run_id"])
    materialized = wizard.materialize(run["run_id"], "mls_rich")

    manifest_csv = Path(manifest["files"]["label_manifest_csv"])
    manifest_rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
    assert manifest_rows[0]["physical_object_id"] == "obj_0002"
    assert manifest_rows[0]["raw_label"] == "Ahuevada"
    assert manifest_rows[0]["label_policy"] == "include"
    assert manifest_rows[0]["include"] == "True"
    assert "extra_fields" in manifest_rows[0]

    memberships = DatasetService(settings.data_dir).list_ml_set_memberships(dataset_id="d1", ml_set_id="mls_rich")
    membership = next(row for row in memberships if row["take_id"] == "2026-05-25T150423_025")
    assert membership["physical_object_id"] == "obj_0002"
    assert membership["split"] == "unassigned"
    assert membership["raw_label"] == "Ahuevada"
    assert membership["label_policy"] == "include"
    assert membership["extra_fields"] == {"note": "keep"}
    review_membership = next(row for row in memberships if row["take_id"] == "2026-05-29T221714_022")
    assert review_membership["include"] is False
    assert materialized["validation_report"]["trainable_count"] == 2


def test_assign_splits_keeps_same_physical_object_together_after_materialization(tmp_path: Path) -> None:
    settings, wizard, run = _setup_run(tmp_path)
    wizard.ingest_table(run["run_id"], content=RICH_TSV, input_format="tsv")
    wizard.reconcile(run["run_id"])
    wizard.materialize(run["run_id"], "mls_rich")
    service = DatasetService(settings.data_dir)
    result = service.assign_ml_set_splits(ml_set_id="mls_rich", dataset_id="d1", dry_run=True)
    grouped_assignments = {item["group_key"]: item["take_ids"] for item in result["assignments"]}
    assert grouped_assignments["obj_0002"] == ["2026-05-25T150423_025", "2026-05-25T150423_026"]
