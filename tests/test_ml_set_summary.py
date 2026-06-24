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
