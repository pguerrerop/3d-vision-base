from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from vision_3d_acquisition.acquisition.replay_dataset import ReplayableAcquisitionService
from vision_3d_acquisition.acquisition.trispector_25d import TriSpectorFtpAcquisitionAdapter
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.processes.bindings import ProcessBindingService


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


def _build_trispector_raw_image(path: Path) -> None:
    h = 128
    w = 160
    reflectance = np.full((h, w), 120, dtype=np.uint8)
    height = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    low = (height & 0xFF).astype(np.uint8)
    high = ((height >> 8) & 0xFF).astype(np.uint8)
    payload = np.empty((h * 2, w), dtype=np.uint8)
    payload[0::2, :] = low
    payload[1::2, :] = high
    composite = np.vstack([reflectance, payload])
    Image.fromarray(composite).save(path)


def _write_recipe_for_pipeline(settings: ApiSettings, pipeline_id: str) -> str:
    recipe_id = f"recipe_{pipeline_id}"
    path = settings.data_dir / "processes" / "recipes" / f"{recipe_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": recipe_id,
        "pipeline_instance_id": f"instance_{pipeline_id}",
        "source_run_id": None,
        "type": "production",
        "notes": None,
        "created_at": datetime.now(UTC).isoformat(),
        "template_id": "manual_pipeline_binding",
        "pipeline_snapshot": {"pipeline_id": pipeline_id},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return recipe_id


def test_trispector_upload_writes_replay_manifest_and_is_immutable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    upload = tmp_path / "upload.png"
    _build_trispector_raw_image(upload)

    adapter = TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0")
    first = adapter.parse_and_register_trispector_upload(upload, take_id="take_replay_001")
    take_dir = settings.incoming_dir / "take_replay_001"
    manifest_path = take_dir / "replay_manifest.json"

    assert manifest_path.is_file()
    manifest_a = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_a["take_id"] == "take_replay_001"
    assert manifest_a["assets"]["heightmap"] == "height16.tif"
    assert manifest_a["immutable"] is True
    assert first["replay_manifest"]["take_id"] == "take_replay_001"

    service = ReplayableAcquisitionService(settings.data_dir)
    manifest_b = service.write_replay_manifest(
        take_dir=take_dir,
        take_id="take_replay_001",
        source_id="trispector_ftp_0",
        metadata=json.loads((take_dir / "metadata.json").read_text(encoding="utf-8")),
        parser_metadata={},
        session_ref=None,
    )
    assert manifest_a == manifest_b


def test_active_replay_session_auto_attaches_trispector_take(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    datasets = DatasetService(settings.data_dir)
    datasets.create_dataset(dataset_id="d1", name="Dataset 1")
    datasets.create_session(dataset_id="d1", session_id="s1", name="Session 1")

    replay = ReplayableAcquisitionService(settings.data_dir)
    replay.start_session(dataset_id="d1", session_id="s1", metadata={"campaign": "batch A"})

    upload = tmp_path / "upload_attach.png"
    _build_trispector_raw_image(upload)
    TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0").parse_and_register_trispector_upload(upload, take_id="take_replay_002")

    take_meta = datasets.load_take_metadata(take_id="take_replay_002", source_metadata={})
    manifest = json.loads((settings.incoming_dir / "take_replay_002" / "replay_manifest.json").read_text(encoding="utf-8"))
    assert take_meta["dataset_id"] == "d1"
    assert take_meta["session_id"] == "s1"
    assert manifest["dataset_id"] == "d1"
    assert manifest["session_id"] == "s1"


def test_replay_take_uses_binding_pipeline_execution_path(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    recipe_id = _write_recipe_for_pipeline(settings, "mining_steel_ball_classification_25d")
    ProcessBindingService(settings).create_binding(
        name="TriSpector 25D binding",
        source_id="trispector_ftp_0",
        modality="heightmap",
        purpose="acquisition_inspection",
        pipeline_id="mining_steel_ball_classification_25d",
        active_recipe_version_id=recipe_id,
        calibration_profile_id=None,
        enabled=True,
    )
    upload = tmp_path / "upload_replay.png"
    _build_trispector_raw_image(upload)
    TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0").parse_and_register_trispector_upload(upload, take_id="take_replay_003")

    payload = ReplayableAcquisitionService(settings.data_dir).replay_take(take_id="take_replay_003")
    assert payload["pipeline_id"] == "mining_steel_ball_classification_25d"
    assert payload["result"]["ok"] is True
    assert (settings.processed_dir / "take_replay_003" / "DONE").is_file()


def test_replay_session_replays_manifest_takes_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    datasets = DatasetService(settings.data_dir)
    datasets.create_dataset(dataset_id="d1", name="Dataset 1")
    datasets.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    replay = ReplayableAcquisitionService(settings.data_dir)
    replay.start_session(dataset_id="d1", session_id="s1")

    upload = tmp_path / "upload_session.png"
    _build_trispector_raw_image(upload)
    TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0").parse_and_register_trispector_upload(upload, take_id="take_replay_004")

    result = replay.replay_session(session_id="s1", dataset_id="d1", pipeline_id="mining_steel_ball_classification_25d")
    assert result["count"] == 1
    assert result["replayed_takes"][0]["take_id"] == "take_replay_004"
