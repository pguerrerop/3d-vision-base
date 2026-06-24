from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from vision_3d_acquisition.api.processing_jobs import (
    ProcessingJobCreateRequest,
    ProcessingJobService,
    resolve_processing_job_take_ids,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


def make_settings(data_dir: Path) -> ApiSettings:
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


def create_take(settings: ApiSettings, *, dataset_id: str, session_id: str, take_id: str) -> None:
    dataset_service = DatasetService(settings.data_dir)
    if dataset_service.get_dataset(dataset_id) is None:
        dataset_service.create_dataset(dataset_id=dataset_id, name=dataset_id)
    if dataset_service.get_session(dataset_id, session_id) is None:
        dataset_service.create_session(dataset_id=dataset_id, session_id=session_id, name=session_id)

    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "take_id": take_id,
        "created_at": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "dataset_id": dataset_id,
    }
    (take_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (take_dir / "READY").touch()
    dataset_service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        updates={"friendly_name": take_id},
        source_metadata=metadata,
    )


def wait_for_job(service: ProcessingJobService, job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = service.load_job(job_id)
        if job and job.status in {"completed", "failed", "cancelled", "partial"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_resolve_processing_job_take_ids_by_dataset_and_session(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    create_take(settings, dataset_id="d1", session_id="s1", take_id="take_a")
    create_take(settings, dataset_id="d1", session_id="s2", take_id="take_b")

    assert resolve_processing_job_take_ids(settings, dataset_id="d1", dataset_session_id="s1") == ["take_a"]
    assert resolve_processing_job_take_ids(settings, dataset_id="d1") == ["take_a", "take_b"]


def test_processing_job_service_executes_and_persists_progress(monkeypatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    create_take(settings, dataset_id="d1", session_id="s1", take_id="take_a")

    def _fake_dispatch(**kwargs):
        return {
            "ok": True,
            "take_id": kwargs["take_id"],
            "pipeline_id": kwargs["pipeline_id"],
            "status": "completed",
            "run_id": f"run_{kwargs['take_id']}",
        }

    monkeypatch.setattr("vision_3d_acquisition.api.processing_jobs.dispatch_take_processing", _fake_dispatch)

    service = ProcessingJobService(settings)
    job = service.create_job(
        ProcessingJobCreateRequest(
            take_ids=["take_a"],
            pipeline_id="3d_ball_inspection",
            created_by="tests",
        )
    )
    finished = wait_for_job(service, job.id)

    assert finished.status == "completed"
    assert finished.completed_takes == 1
    assert finished.failed_takes == 0
    assert finished.take_items[0].run_id == "run_take_a"
    assert finished.summary.get("scope_label") == "Selected takes"
    assert (settings.data_dir / "runtime" / "processing_jobs" / f"{job.id}.json").is_file()
    assert (settings.data_dir / "runtime" / "processing_jobs" / f"{job.id}.jsonl").is_file()
