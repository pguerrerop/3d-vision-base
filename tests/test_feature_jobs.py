from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_3d_acquisition.api.feature_jobs import FeatureJobRequest, FeatureJobService
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml.features.surface_sphere_fit_workflow import backfill_sidecar_name
from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, save_heightmap_npz
import numpy as np


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


def _write_old_style_take(data_dir: Path, take_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "metadata.json").write_text(json.dumps({"take_id": take_id, "created_at": "2026-05-28T12:00:00Z"}), encoding="utf-8")
    (take_dir / "READY").touch()
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "status": "ok",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "feature_eccentricity": 0.2,
                        "bbox_px": [40, 40, 80, 80],
                        "point_count": 120,
                    }
                ],
                "files": {"normalized_heightmap": "normalized_heightmap.npz"},
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "contour.json").write_text(
        json.dumps({"objects": [{"object_id": 1, "contour_px": [[40, 40], [120, 40], [120, 120], [40, 120]]}]}),
        encoding="utf-8",
    )
    y, x = np.mgrid[0:160, 0:160]
    z = np.sqrt(np.maximum(0.0, 40.0**2 - (x - 80.0) ** 2 - (y - 80.0) ** 2)).astype(np.float32)
    frame = HeightmapFrame(
        z_mm=z,
        valid_mask=z > 0.0,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="belt_plane_normalized_mm",
    )
    save_heightmap_npz(frame, result_dir / "normalized_heightmap.npz")
    (result_dir / "DONE").touch()


def test_feature_job_persists_incremental_progress(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = _settings(data_dir)
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="ml_set_1", name="ML Set", task_type="classification")
    for take_id in ("take_a", "take_b"):
        _write_old_style_take(data_dir, take_id)
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"created_at": "2026-05-28T12:00:00Z"})
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="ml_set_1", take_id=take_id)

    observed_completed: list[int] = []

    def _fake_dispatch(*, settings, take_id, pipeline_id, **kwargs):  # noqa: ANN001
        job_files = sorted((settings.data_dir / "runtime" / "feature_jobs").glob("fj_*.json"))
        if job_files:
            payload = json.loads(job_files[-1].read_text(encoding="utf-8"))
            observed_completed.append(int(payload.get("completed_takes") or 0))
        return {"status": "ok", "take_id": take_id, "pipeline_id": pipeline_id}

    monkeypatch.setattr("vision_3d_acquisition.api.feature_jobs.dispatch_take_processing", _fake_dispatch)

    job_service = FeatureJobService(settings)
    job = job_service.create_job(
        FeatureJobRequest(
            scope="ml_set",
            dataset_id="d1",
            ml_set_id="ml_set_1",
            feature_key="surface_sphere_fit_rmse_mm",
            mode="reprocess",
            pipeline_id="mining_steel_ball_classification_25d",
        )
    )
    job_service._active_threads[job.id].join(timeout=5)
    finished = job_service.load_job(job.id)
    assert finished is not None
    assert finished.completed_takes == 2
    assert observed_completed, "expected incremental progress snapshots while takes were finishing"
    assert max(observed_completed) >= 1


def test_feature_job_reprocess_resolves_ml_set_takes(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = _settings(data_dir)
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="ml_set_1", name="ML Set", task_type="classification")
    _write_old_style_take(data_dir, "take_a")
    service.upsert_take_metadata(take_id="take_a", dataset_id="d1", session_id="s1", updates={}, source_metadata={"created_at": "2026-05-28T12:00:00Z"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="ml_set_1", take_id="take_a")

    dispatched: list[str] = []

    def _fake_dispatch(*, settings, take_id, pipeline_id, **kwargs):  # noqa: ANN001
        dispatched.append(take_id)
        return {"status": "ok", "take_id": take_id, "pipeline_id": pipeline_id}

    monkeypatch.setattr("vision_3d_acquisition.api.feature_jobs.dispatch_take_processing", _fake_dispatch)

    job_service = FeatureJobService(settings)
    job = job_service.create_job(
        FeatureJobRequest(
            scope="ml_set",
            dataset_id="d1",
            ml_set_id="ml_set_1",
            feature_key="surface_sphere_fit_rmse_mm",
            mode="reprocess",
            pipeline_id="mining_steel_ball_classification_25d",
        )
    )
    assert job.total_takes == 1
    job_service._active_threads[job.id].join(timeout=5)
    finished = job_service.load_job(job.id)
    assert finished is not None
    assert finished.completed_takes == 1
    assert dispatched == ["take_a"]


def test_feature_job_backfill_writes_sidecar_provenance(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = _settings(data_dir)
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="ml_set_1", name="ML Set", task_type="classification")
    _write_old_style_take(data_dir, "take_a")
    service.upsert_take_metadata(take_id="take_a", dataset_id="d1", session_id="s1", updates={}, source_metadata={"created_at": "2026-05-28T12:00:00Z"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="ml_set_1", take_id="take_a")

    job_service = FeatureJobService(settings)
    job = job_service.create_job(
        FeatureJobRequest(
            scope="ml_set",
            dataset_id="d1",
            ml_set_id="ml_set_1",
            feature_key="surface_sphere_fit_rmse_mm",
            mode="backfill",
            only_missing=True,
        )
    )
    job_service._active_threads[job.id].join(timeout=5)
    finished = job_service.load_job(job.id)
    assert finished is not None
    assert finished.completed_takes == 1
    sidecar = data_dir / "processed" / "take_a" / backfill_sidecar_name("surface_sphere_fit_rmse_mm")
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["source"] == "feature_backfill"
    assert payload["entries"][0]["source"] == "feature_backfill"


def test_feature_job_reprocess_accepts_sphere_consistency_keys(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = _settings(data_dir)
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="ml_set_1", name="ML Set", task_type="classification")
    _write_old_style_take(data_dir, "take_a")
    service.upsert_take_metadata(take_id="take_a", dataset_id="d1", session_id="s1", updates={}, source_metadata={"created_at": "2026-05-28T12:00:00Z"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="ml_set_1", take_id="take_a")

    monkeypatch.setattr(
        "vision_3d_acquisition.api.feature_jobs.dispatch_take_processing",
        lambda **kwargs: {"status": "ok"},
    )

    job_service = FeatureJobService(settings)
    job = job_service.create_job(
        FeatureJobRequest(
            scope="ml_set",
            dataset_id="d1",
            ml_set_id="ml_set_1",
            feature_key="surface_sphere_fit_residual_mad_norm",
            mode="reprocess",
            pipeline_id="mining_steel_ball_classification_25d",
        )
    )
    job_service._active_threads[job.id].join(timeout=5)
    finished = job_service.load_job(job.id)
    assert finished is not None
    assert finished.completed_takes == 1


def test_feature_job_invalid_feature_key(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    job_service = FeatureJobService(settings)
    with pytest.raises(ValueError, match="unsupported feature_key for backfill"):
        job_service.create_job(
            FeatureJobRequest(
                scope="takes",
                take_ids=["take_a"],
                feature_key="not_a_real_feature",
                mode="backfill",
            )
        )
    with pytest.raises(ValueError, match="unsupported feature_key for reprocess"):
        job_service.create_job(
            FeatureJobRequest(
                scope="takes",
                take_ids=["take_a"],
                feature_key="feature_eccentricity",
                mode="reprocess",
                pipeline_id="mining_steel_ball_classification_25d",
            )
        )


def test_feature_job_missing_ml_set(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    job_service = FeatureJobService(settings)
    with pytest.raises(ValueError):
        job_service.create_job(
            FeatureJobRequest(
                scope="ml_set",
                ml_set_id="missing_set",
                feature_key="surface_sphere_fit_rmse_mm",
                mode="backfill",
            )
        )


def test_feature_job_status_endpoint_shape(monkeypatch, tmp_path: Path) -> None:
    from vision_3d_acquisition.api.main import create_feature_job, get_feature_job

    data_dir = tmp_path / "data"
    settings = _settings(data_dir)
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="ml_set_1", name="ML Set", task_type="classification")
    _write_old_style_take(data_dir, "take_a")
    service.upsert_take_metadata(take_id="take_a", dataset_id="d1", session_id="s1", updates={}, source_metadata={"created_at": "2026-05-28T12:00:00Z"})
    service.add_take_to_ml_set(dataset_id="d1", ml_set_id="ml_set_1", take_id="take_a")

    monkeypatch.setattr(
        "vision_3d_acquisition.api.feature_jobs.dispatch_take_processing",
        lambda **kwargs: {"status": "ok"},
    )

    created = create_feature_job(
        FeatureJobRequest(
            scope="ml_set",
            dataset_id="d1",
            ml_set_id="ml_set_1",
            feature_key="surface_sphere_fit_rmse_mm",
            mode="reprocess",
            pipeline_id="mining_steel_ball_classification_25d",
        ),
        settings=settings,
    )
    job_id = created["job_id"]
    payload = get_feature_job(job_id, settings=settings)
    assert payload["job_id"] == job_id
    assert payload["job"]["feature_key"] == "surface_sphere_fit_rmse_mm"


def test_a_job_stays_readable_while_its_worker_rewrites_it(tmp_path: Path) -> None:
    """Polling a running job must never report it as unknown.

    The worker rewrites the record as it progresses while callers poll status.
    A plain write truncates the file first, so a poll landing inside that window
    read nothing, failed to parse, and surfaced as 404 "Unknown feature job" for
    a job that was running perfectly well.
    """
    import threading

    from vision_3d_acquisition.api.feature_jobs import FeatureJobRecord

    settings = _settings(tmp_path / "data")
    service = FeatureJobService(settings)

    job = FeatureJobRecord(
        id="fj_concurrent",
        created_at="2026-05-28T12:00:00Z",
        status="running",
        scope="ml_set",
        dataset_id="d1",
        ml_set_id="ml_set_1",
        feature_key="surface_sphere_fit_rmse_mm",
        feature_display_name="Surface sphere fit RMSE",
        mode="reprocess",
        total_takes=1,
    )
    service._write_job(job)

    stop = threading.Event()
    missed: list[str] = []

    def rewrite() -> None:
        counter = 0
        while not stop.is_set():
            counter += 1
            service._write_job(job.model_copy(update={"completed_takes": counter}))

    writer = threading.Thread(target=rewrite, daemon=True)
    writer.start()
    try:
        for _ in range(400):
            if service.load_job("fj_concurrent") is None:
                missed.append("read returned no record while the job existed")
                break
    finally:
        stop.set()
        writer.join(timeout=5)

    assert missed == [], missed[0] if missed else ""
