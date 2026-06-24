from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.physical_objects import PhysicalObjectRegistry
from vision_3d_acquisition.ml.ingestion_wizard import IngestionWizardService


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
    (settings.processed_dir / take_id).mkdir(parents=True, exist_ok=True)
    (settings.processed_dir / take_id / "result.json").write_text(json.dumps({"summary": {"decision": "accept"}}), encoding="utf-8")


def test_registry_sync_from_ingestion_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    _write_take(settings, "take_001", "s1")
    _write_take(settings, "take_002", "s1")
    service.upsert_take_metadata(take_id="take_001", dataset_id="d1", session_id="s1", updates={"physical_object_id": "object_000001", "expected_class": "BALL_GOOD"}, source_metadata={})
    service.upsert_take_metadata(take_id="take_002", dataset_id="d1", session_id="s1", updates={"physical_object_id": "object_000001", "expected_class": "BALL_GOOD"}, source_metadata={})

    registry = PhysicalObjectRegistry(settings)
    synced = registry.sync_from_manifest_rows("d1", [
        {"physical_object_id": "object_000001", "take_id": "take_001", "source_session_id": "s1", "raw_operator_label": "Buena", "normalized_class": "BALL_GOOD", "superclass": "BALL", "annotation_confidence": 0.91, "source_row_index": 1, "d1_mm": 84, "d2_mm": 85, "d3_mm": 84},
        {"physical_object_id": "object_000001", "take_id": "take_002", "source_session_id": "s1", "raw_operator_label": "Buena", "normalized_class": "BALL_GOOD", "superclass": "BALL", "annotation_confidence": 0.91, "source_row_index": 1, "d1_mm": 84, "d2_mm": 85, "d3_mm": 84},
    ])

    assert len(synced) == 1
    payload = registry.get_object("d1", "object_000001")
    assert payload is not None
    assert payload["normalized_class"] == "BALL_GOOD"
    assert payload["observation_take_ids"] == ["take_001", "take_002"]


def test_ingestion_reconcile_creates_physical_objects(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    _write_take(settings, "2026-05-25T183433_038", "runtime")
    service.upsert_take_metadata(take_id="2026-05-25T183433_038", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "runtime"})

    wizard = IngestionWizardService(settings.data_dir)
    run = wizard.create_run(dataset_id="d1", session_ids=["s1"], name="test")
    wizard.ingest_table(run["run_id"], content="label\timage_ref\td1\td2\td3\nBuena\t38\t84\t85\t84\n", input_format="tsv")
    wizard.reconcile(run["run_id"])

    payload = PhysicalObjectRegistry(settings).list_objects("d1")
    assert len(payload) == 1
    assert payload[0]["normalized_class"] == "BALL_GOOD"
