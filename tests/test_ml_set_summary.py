from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml.ml_set_summary import MLSetSummaryService


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


def _write_take(settings: ApiSettings, take_id: str, created_at: str = "2026-06-03T12:00:00Z") -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "metadata.json").write_text(json.dumps({"take_id": take_id, "created_at": created_at}), encoding="utf-8")


def test_ml_set_summary_detects_leakage_and_class_distribution(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="ds1", name="Dataset 1")
    service.create_session(dataset_id="ds1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="ds1", ml_set_id="mls1", name="MLS 1", task_type="classification")

    _write_take(settings, "t1")
    _write_take(settings, "t2")
    service.upsert_take_metadata(take_id="t1", dataset_id="ds1", session_id="s1", updates={"expected_class": "BALL_GOOD", "superclass_labels": ["BALL"]}, source_metadata={})
    service.upsert_take_metadata(take_id="t2", dataset_id="ds1", session_id="s1", updates={"expected_class": "BALL_GOOD", "superclass_labels": ["BALL"]}, source_metadata={})
    service.add_take_to_ml_set(dataset_id="ds1", ml_set_id="mls1", take_id="t1", split="train", physical_object_id="object_1", expected_class="BALL_GOOD")
    service.add_take_to_ml_set(dataset_id="ds1", ml_set_id="mls1", take_id="t2", split="validation", physical_object_id="object_1", expected_class="BALL_GOOD")

    summary = MLSetSummaryService(settings).build_summary("mls1", "ds1")

    assert summary["identity"]["take_count"] == 2
    assert summary["class_distribution"]["BALL_GOOD"] == 2
    assert summary["split_summary"]["leaked_object_ids"] == ["object_1"]


def test_ml_set_member_listing_supports_scope_filters(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="ds1", name="Dataset 1")
    service.create_session(dataset_id="ds1", session_id="s1", name="Session 1")
    service.create_session(dataset_id="ds1", session_id="s2", name="Session 2")
    service.create_ml_set(dataset_id="ds1", ml_set_id="mls1", name="MLS 1", task_type="classification")

    _write_take(settings, "t1")
    _write_take(settings, "t2")
    _write_take(settings, "t3")
    service.upsert_take_metadata(
        take_id="t1",
        dataset_id="ds1",
        session_id="s1",
        updates={"expected_class": "BALL_GOOD", "superclass_labels": ["BALL"], "validation_status": "valid"},
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="t2",
        dataset_id="ds1",
        session_id="s1",
        updates={"expected_class": "BALL_BAD", "superclass_labels": ["BALL_DEFECT"], "validation_status": "needs_review"},
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="t3",
        dataset_id="ds1",
        session_id="s2",
        updates={"expected_class": "SCRAP_BOLT", "superclass_labels": ["SCRAP"], "validation_status": "unreviewed"},
        source_metadata={},
    )
    service.add_take_to_ml_set(
        dataset_id="ds1",
        ml_set_id="mls1",
        take_id="t1",
        split="train",
        physical_object_id="obj_a",
        expected_class="BALL_GOOD",
        expected_subclass="BALL",
        raw_label="Operador Bueno",
        default_trainable=True,
        review_required=False,
    )
    service.add_take_to_ml_set(
        dataset_id="ds1",
        ml_set_id="mls1",
        take_id="t2",
        split="unassigned",
        physical_object_id="obj_a",
        expected_class="BALL_BAD",
        expected_subclass="BALL_DEFECT",
        raw_label="Operador Dudoso",
        default_trainable=False,
        review_required=True,
    )
    service.add_take_to_ml_set(
        dataset_id="ds1",
        ml_set_id="mls1",
        take_id="t3",
        split="validation",
        physical_object_id="obj_b",
        expected_class="SCRAP_BOLT",
        expected_subclass="SCRAP",
        raw_label="Chatarra",
        default_trainable=True,
        review_required=False,
    )

    payload = MLSetSummaryService(settings).list_members(
        "mls1",
        "ds1",
        membership_status="review_required",
        session_id="s1",
    )

    assert payload["filtered_count"] == 1
    assert payload["stats"]["object_count"] == 1
    row = payload["items"][0]
    assert row["take_id"] == "t2"
    assert row["raw_label"] == "Operador Dudoso"
    assert row["normalized_class"] == "BALL_BAD"
    assert row["superclass"] == "BALL_DEFECT"
    assert row["split"] == "unassigned"
    assert row["review_required"] is True
    assert "missing_split" in row["warnings"]


def test_ml_set_member_facets_cover_full_scope_counts(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="ds1", name="Dataset 1")
    service.create_session(dataset_id="ds1", session_id="s1", name="Session 1")
    service.create_session(dataset_id="ds1", session_id="s2", name="Session 2")
    service.create_ml_set(dataset_id="ds1", ml_set_id="mls1", name="MLS 1", task_type="classification")

    _write_take(settings, "t1")
    _write_take(settings, "t2")
    _write_take(settings, "t3")
    service.upsert_take_metadata(
        take_id="t1",
        dataset_id="ds1",
        session_id="s1",
        updates={"expected_class": "BALL_GOOD", "superclass_labels": ["BALL"], "validation_status": "valid"},
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="t2",
        dataset_id="ds1",
        session_id="s1",
        updates={"expected_class": "BALL_BAD", "superclass_labels": ["BALL_DEFECT"], "validation_status": "needs_review"},
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="t3",
        dataset_id="ds1",
        session_id="s2",
        updates={"expected_class": "SCRAP_BOLT", "superclass_labels": ["SCRAP"], "validation_status": "unreviewed"},
        source_metadata={},
    )
    service.add_take_to_ml_set(
        dataset_id="ds1",
        ml_set_id="mls1",
        take_id="t1",
        split="train",
        physical_object_id="obj_2",
        expected_class="BALL_GOOD",
        expected_subclass="BALL",
        raw_label="Operador Bueno",
        default_trainable=True,
    )
    service.add_take_to_ml_set(
        dataset_id="ds1",
        ml_set_id="mls1",
        take_id="t2",
        split="unassigned",
        physical_object_id="obj_2",
        expected_class="BALL_BAD",
        expected_subclass="BALL_DEFECT",
        raw_label="Operador Dudoso",
        review_required=True,
    )
    service.add_take_to_ml_set(
        dataset_id="ds1",
        ml_set_id="mls1",
        take_id="t3",
        split="validation",
        physical_object_id="obj_10",
        expected_class="SCRAP_BOLT",
        expected_subclass="SCRAP",
        raw_label="Chatarra",
        default_trainable=True,
    )

    facets = MLSetSummaryService(settings).build_member_facets("mls1", "ds1", session_id="s1")

    assert facets["stats"]["member_count"] == 2
    assert [item["value"] for item in facets["object_ids"]] == ["obj_2", "obj_10"]
    assert facets["object_ids"][0]["take_count"] == 2
    assert {item["value"]: item["count"] for item in facets["raw_labels"]} == {
        "Operador Bueno": 1,
        "Operador Dudoso": 1,
    }
    assert {item["value"]: item["count"] for item in facets["splits"]} == {
        "train": 1,
        "unassigned": 1,
        "validation": 1,
    }
    assert {item["value"]: item["count"] for item in facets["sessions"]} == {
        "s1": 2,
        "s2": 1,
    }
    membership_counts = {item["value"]: item["count"] for item in facets["membership_statuses"]}
    assert membership_counts["all"] == 3
    assert membership_counts["uncertain"] == 1


def test_export_file_falls_back_to_the_catalog_without_materialized_artifacts(tmp_path: Path) -> None:
    """label_schema/tasks/snapshot used to reference an unassigned `base`

    variable when the materialized file was missing on disk — a leftover from
    before ml_set.json became a row in the catalog. Every call with nothing
    materialized yet raised NameError instead of serving anything.
    """
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="ds1", name="Dataset 1")
    service.create_session(dataset_id="ds1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="ds1", ml_set_id="mls1", name="MLS 1", task_type="classification")
    _write_take(settings, "t1")
    service.upsert_take_metadata(take_id="t1", dataset_id="ds1", session_id="s1", updates={}, source_metadata={})
    service.add_take_to_ml_set(dataset_id="ds1", ml_set_id="mls1", take_id="t1", split="train")

    summary = MLSetSummaryService(settings)

    for kind in ("manifest", "splits", "label_schema", "tasks", "snapshot"):
        path = summary.export_file("mls1", kind, dataset_id="ds1")
        assert path.is_file(), f"{kind} did not produce a readable file"
