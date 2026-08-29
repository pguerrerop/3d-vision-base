from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.api.main import (
    runtime_worker_events,
    runtime_worker_status,
    runtime_worker_stop_request,
    runtime_workers,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.processes.bindings import ProcessBindingService
from vision_3d_acquisition.runtime.worker_manager import RuntimeWorkerManager
from vision_3d_acquisition.runtime.workers import (
    AcquisitionPublicationWorker,
    FusionPublisherWorker,
    HeightmapAcquisitionProcessingWorker,
    RgbAcquisitionProcessingWorker,
)


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


def _write_take(settings: ApiSettings, take_id: str, *, source: str, modalities: list[str], group_id: str | None = None) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    if "rgb" in modalities:
        files["rgb"] = "rgb.png"
        (take_dir / "rgb.png").write_bytes(b"rgb")
    if "heightmap" in modalities:
        files["heightmap"] = "heightmap.npz"
        (take_dir / "heightmap.npz").write_bytes(b"hm")
    payload = {
        "take_id": take_id,
        "source": source,
        "created_at": datetime.now(UTC).isoformat(),
        "modalities": modalities,
        "files": files,
    }
    if group_id:
        payload["acquisition_group_id"] = group_id
    (take_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    (take_dir / "READY").touch()


def _write_processed_result(settings: ApiSettings, take_id: str, *, family: str, object_candidates: list[dict]) -> None:
    out = settings.processed_dir / take_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "status": "ok",
                "processing_pipeline": {"pipeline_family": family},
                "object_candidates": object_candidates,
            }
        ),
        encoding="utf-8",
    )
    (out / "DONE").touch()


def _write_recipe(settings: ApiSettings, *, recipe_id: str, pipeline_id: str) -> None:
    recipes_dir = settings.data_dir / "processes" / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    (recipes_dir / f"{recipe_id}.json").write_text(
        json.dumps(
            {
                "id": recipe_id,
                "pipeline_instance_id": "pipeline_instance_demo",
                "source_run_id": None,
                "type": "production",
                "notes": None,
                "created_at": datetime.now(UTC).isoformat(),
                "template_id": "runtime_worker_recipe",
                "pipeline_snapshot": {"pipeline_id": pipeline_id},
            }
        ),
        encoding="utf-8",
    )


def _create_binding(settings: ApiSettings, *, source_id: str, modality: str, purpose: str, pipeline_id: str, recipe_id: str) -> dict:
    _write_recipe(settings, recipe_id=recipe_id, pipeline_id=pipeline_id)
    return ProcessBindingService(settings).create_binding(
        name=f"binding_{source_id}_{modality}_{purpose}",
        source_id=source_id,
        modality=modality,
        purpose=purpose,  # type: ignore[arg-type]
        pipeline_id=pipeline_id,
        active_recipe_version_id=recipe_id,
        calibration_profile_id=None,
        enabled=True,
    )


def test_worker_registration_status_heartbeat_and_events_persisted(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)

    manager.register_worker(worker_id="w1", worker_type="rgb_worker", source_id="usb_camera_0")
    manager.heartbeat("w1", "running", {"step": "poll"})
    manager.append_worker_event("w1", {"worker_id": "w1", "event_type": "CUSTOM", "message": "hello", "severity": "info"})

    status = manager.get_worker_status("w1")
    assert status is not None
    assert status["worker_id"] == "w1"
    assert status["last_heartbeat"] is not None
    events = manager.get_worker_events("w1")
    assert any(item["event_type"] == "CUSTOM" for item in events)
    assert (settings.data_dir / "runtime" / "workers" / "events" / "w1.jsonl").is_file()


def test_rgb_worker_once_processes_eligible_take_using_binding(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)
    _write_take(settings, "take_rgb_001", source="usb_camera_0", modalities=["rgb"])
    _create_binding(
        settings,
        source_id="usb_camera_0",
        modality="rgb",
        purpose="acquisition_inspection",
        pipeline_id="mining_steel_ball_classification_2d",
        recipe_id="recipe_rgb_worker",
    )

    def fake_dispatch(*_args, **kwargs):
        return {"ok": True, "result": {"config_snapshot_hash": "cfg_rgb"}, "pipeline_id": kwargs.get("pipeline_id")}

    monkeypatch.setattr("vision_3d_acquisition.acquisition.process_integration.dispatch_take_processing", fake_dispatch)

    worker = RgbAcquisitionProcessingWorker(
        settings=settings,
        manager=manager,
        worker_id="rgb_worker_test",
        source_id="usb_camera_0",
        station_id="station_rgb",
        poll_interval_sec=0.01,
    )
    summary = worker.run(once=True, dry_run=False)

    assert summary["processed_count"] == 1
    assert (settings.data_dir / "runtime" / "queues" / "completed_processing.jsonl").is_file()
    metadata = json.loads((settings.incoming_dir / "take_rgb_001" / "metadata.json").read_text(encoding="utf-8"))
    assert str(metadata.get("acquisition_group_id") or "").startswith("ag_")


def test_25d_worker_once_processes_eligible_take_using_binding(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)
    _write_take(settings, "take_25d_001", source="trispector_ftp_0", modalities=["heightmap"])
    _create_binding(
        settings,
        source_id="trispector_ftp_0",
        modality="heightmap",
        purpose="acquisition_inspection",
        pipeline_id="mining_steel_ball_classification_25d",
        recipe_id="recipe_25d_worker",
    )

    def fake_dispatch(*_args, **kwargs):
        return {"ok": True, "result": {"config_snapshot_hash": "cfg_25d"}, "pipeline_id": kwargs.get("pipeline_id")}

    monkeypatch.setattr("vision_3d_acquisition.acquisition.process_integration.dispatch_take_processing", fake_dispatch)

    worker = HeightmapAcquisitionProcessingWorker(
        settings=settings,
        manager=manager,
        worker_id="worker_25d_test",
        source_id="trispector_ftp_0",
        station_id="station_25d",
        poll_interval_sec=0.01,
    )
    summary = worker.run(once=True, dry_run=False)
    assert summary["processed_count"] == 1


def test_fusion_publisher_worker_skips_incomplete_group(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g_incomplete", station_id="st_i")
    _write_take(settings, "take_rgb_only", source="usb_camera_0", modalities=["rgb"], group_id=group["id"])
    _write_processed_result(
        settings,
        "take_rgb_only",
        family="2d",
        object_candidates=[{"id": "c2d_1", "centroid_px": [1, 1], "bbox_px": [0, 0, 1, 1], "classification_hints": {"class_label": "Bola buena"}, "confidence": 0.9}],
    )

    worker = FusionPublisherWorker(
        settings=settings,
        manager=manager,
        worker_id="fusion_worker_skip",
        source_id=None,
        station_id=None,
        poll_interval_sec=0.01,
    )
    summary = worker.run(once=True, dry_run=False)
    assert summary["processed_count"] == 0
    assert summary["skipped_count"] >= 1
    assert not (settings.data_dir / "published_inspection_results" / "index.json").is_file()


def test_fusion_publisher_worker_fuses_and_publishes_complete_group(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g_complete", station_id="st_c")
    _write_take(settings, "take_rgb_ok", source="usb_camera_0", modalities=["rgb"], group_id=group["id"])
    _write_take(settings, "take_25d_ok", source="trispector_ftp_0", modalities=["heightmap"], group_id=group["id"])
    _write_processed_result(
        settings,
        "take_rgb_ok",
        family="2d",
        object_candidates=[{"id": "c2d_ok", "centroid_px": [10, 10], "bbox_px": [0, 0, 2, 2], "classification_hints": {"class_label": "Bola buena"}, "confidence": 0.95}],
    )
    _write_processed_result(
        settings,
        "take_25d_ok",
        family="25d",
        object_candidates=[
            {
                "id": "c25d_ok",
                "centroid_px": [10, 10],
                "centroid_world": [1, 2, 3],
                "bbox_px": [0, 0, 2, 2],
                "classification_hints": {"deformation_hint": 0.1, "roundness_hint": 0.9, "flatness_hint": 0.2},
                "measurements": {"height_max_mm": 8.0},
                "confidence": 0.88,
            }
        ],
    )
    _create_binding(
        settings,
        source_id=group["id"],
        modality="acquisition_group",
        purpose="fusion",
        pipeline_id="mining_steel_ball_fusion",
        recipe_id="recipe_fusion_worker",
    )

    worker = FusionPublisherWorker(
        settings=settings,
        manager=manager,
        worker_id="fusion_worker_publish",
        source_id=None,
        station_id=None,
        poll_interval_sec=0.01,
    )
    summary = worker.run(once=True, dry_run=False)

    assert summary["processed_count"] == 1
    index = json.loads((settings.data_dir / "published" / "inspection_results" / "index.json").read_text(encoding="utf-8"))
    assert len(index["recent"]) == 1


def test_worker_stop_request_flag_and_monitoring_api(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)
    manager.register_worker(worker_id="api_worker", worker_type="rgb_worker", source_id="usb_camera_0")
    monkeypatch.setattr("vision_3d_acquisition.api.main.get_runtime_worker_manager", lambda: manager)

    payload_list = runtime_workers()
    payload_one = runtime_worker_status("api_worker")
    payload_stop = runtime_worker_stop_request("api_worker")
    payload_events = runtime_worker_events("api_worker", limit=20)

    assert any(item["worker_id"] == "api_worker" for item in payload_list["workers"])
    assert payload_one["worker_id"] == "api_worker"
    assert payload_stop["stop_requested"] is True
    assert manager.is_stop_requested("api_worker") is True
    assert payload_events["worker_id"] == "api_worker"


def test_publication_worker_generates_operator_summary_from_completed_runs(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    manager = RuntimeWorkerManager(settings)
    queue_dir = settings.data_dir / "runtime" / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "completed_processing.jsonl").write_text(
        json.dumps({"take_id": "take_publish_1", "source_id": "usb_camera_0", "acquisition_group_id": "ag_1"}) + "\n",
        encoding="utf-8",
    )
    out = settings.processed_dir / "take_publish_1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps({"take_id": "take_publish_1", "summary": {"decision": "accept", "confidence": 0.9}, "objects": [{"id": 1}], "artifacts": []}),
        encoding="utf-8",
    )
    worker = AcquisitionPublicationWorker(
        settings=settings,
        manager=manager,
        worker_id="publication_worker_test",
        source_id=None,
        station_id="station_pub",
        poll_interval_sec=0.01,
    )
    summary = worker.run(once=True, dry_run=False)
    assert summary["processed_count"] == 1
    assert (settings.data_dir / "published" / "live_inspection_results" / "index.json").is_file()
