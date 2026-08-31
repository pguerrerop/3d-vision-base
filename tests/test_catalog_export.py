"""The way back out of the catalog.

After stage 2 the labels, validation state, ML set membership and physical
objects exist only as rows. The export is what makes that reversible, so the
test that matters is the round trip: export, then read the result with the
filesystem backend and check every document came back identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.documents import FilesystemDocuments
from vision_3d_acquisition.processing.status_index import append_process_run_index
from vision_3d_acquisition.storage.catalog_export import export_catalog
from vision_3d_acquisition.storage.dataset_store import SqlDocuments


def _populate(settings: ApiSettings, write_take) -> DatasetService:
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo", tags=["a"], notes="una nota")
    service.create_session(dataset_id="demo", session_id="s1", name="Sesion 1", session_type="curated")
    service.create_ml_set(
        dataset_id="demo", ml_set_id="mls", name="Set", task_type="classification"
    )

    write_take("take_a", done=True)
    service.upsert_take_metadata(
        take_id="take_a",
        dataset_id="demo",
        session_id="s1",
        updates={
            "friendly_name": "Bola",
            "tags": ["Wet"],
            "semantic_labels": ["chatarra"],
            "validation_status": "valid",
            "physical_object_id": "obj_1",
            "expected_diameter_mm": 42.5,
            # A key the schema has no column for: it still has to survive.
            "normalization_warnings": ["dudoso"],
        },
    )
    service.add_take_to_ml_set(
        dataset_id="demo",
        ml_set_id="mls",
        take_id="take_a",
        split="train",
        expected_label="chatarra",
        measurements_mm={"d1_mm": 91.5},
    )
    append_process_run_index(
        settings.data_dir,
        take_id="take_a",
        pipeline_instance_id="pipeline_1",
        run_id="run_1",
        pipeline_family="2d",
        status="success",
        run_dir=settings.data_dir / "processes" / "runs" / "pipeline_1" / "run_1",
        created_at="2026-05-19T11:00:00Z",
    )
    return service


def test_every_document_survives_the_round_trip(
    catalog_settings: ApiSettings, catalog_conn, write_take, tmp_path: Path
) -> None:
    _populate(catalog_settings, write_take)
    destination = tmp_path / "export"

    report = export_catalog(catalog_settings.data_dir, destination)

    assert report.datasets == 1
    assert report.takes == 1
    assert report.memberships == 1
    assert report.process_runs == 1
    assert report.warnings == []

    source = SqlDocuments(catalog_settings.data_dir)
    restored = FilesystemDocuments(destination)

    assert restored.read_dataset("demo") == source.read_dataset("demo")
    assert restored.read_session("demo", "s1") == source.read_session("demo", "s1")
    assert restored.read_ml_set("demo", "mls") == source.read_ml_set("demo", "mls")
    assert restored.read_memberships("demo", "mls") == source.read_memberships("demo", "mls")

    take = restored.read_take("demo", "s1", "take_a")
    assert take == source.read_take("demo", "s1", "take_a")
    assert take["semantic_labels"] == ["chatarra"]
    assert take["validation_status"] == "valid"
    # The field no column knows about came through the export too.
    assert take["normalization_warnings"] == ["dudoso"]


def test_the_export_is_readable_as_a_data_directory(
    catalog_settings: ApiSettings, catalog_conn, write_take, tmp_path: Path, monkeypatch
) -> None:
    """The exit: point a data directory at the export and read it without SQLite."""
    _populate(catalog_settings, write_take)
    destination = tmp_path / "export"
    export_catalog(catalog_settings.data_dir, destination)

    monkeypatch.setenv("SENSOR_STUDIO_INDEX", "off")
    service = DatasetService(destination)

    assert [item["id"] for item in service.list_datasets()] == ["demo"]
    assert service.resolve_all_take_memberships("take_a") == [("demo", "s1")]
    assert service.load_take_metadata(take_id="take_a")["validation_status"] == "valid"
    assert [row["take_id"] for row in service.list_ml_set_memberships(dataset_id="demo", ml_set_id="mls")] == [
        "take_a"
    ]


def test_the_export_records_how_to_restore_it(
    catalog_settings: ApiSettings, catalog_conn, write_take, tmp_path: Path
) -> None:
    _populate(catalog_settings, write_take)
    destination = tmp_path / "export"
    export_catalog(catalog_settings.data_dir, destination)

    manifest = json.loads((destination / "EXPORT.json").read_text(encoding="utf-8"))

    assert manifest["counts"]["takes"] == 1
    assert "SENSOR_STUDIO_INDEX=off" in manifest["restore"]
    assert manifest["source"].endswith("data")
    runs = json.loads((destination / "processes" / "index" / "runs.json").read_text(encoding="utf-8"))
    assert [entry["run_id"] for entry in runs["entries"]] == ["run_1"]
