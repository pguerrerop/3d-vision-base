from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from vision_3d_acquisition.acquisition.trispector_25d import TriSpectorFtpAcquisitionAdapter
from vision_3d_acquisition.api.filesystem import get_take_summary
from vision_3d_acquisition.api.settings import ApiSettings
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


def test_trispector_upload_parses_and_registers_assets(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    upload = tmp_path / "upload.png"
    _build_trispector_raw_image(upload)
    adapter = TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0")

    result = adapter.parse_and_register_trispector_upload(upload, take_id="take_trispector_001")
    take_dir = settings.incoming_dir / "take_trispector_001"
    metadata = json.loads((take_dir / "metadata.json").read_text(encoding="utf-8"))

    assert result["take_id"] == "take_trispector_001"
    assert (take_dir / "height16.tif").is_file()
    assert (take_dir / "reflectance.png").is_file()
    assert (take_dir / "parser_metadata.json").is_file()
    assert (take_dir / "raw_upload.png").is_file()
    assert metadata["files"]["heightmap"] == "height16.tif"
    assert metadata["files"]["parser_metadata"] == "parser_metadata.json"
    assert "heightmap" in metadata["modalities"]
    assert result["processing"]["status"] == "processing_binding_missing"
    assert "No active processing binding found" in str(result["processing"].get("warning") or "")


def test_trispector_binding_triggers_25d_processing(tmp_path: Path) -> None:
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
    upload = tmp_path / "upload_active_binding.png"
    _build_trispector_raw_image(upload)
    adapter = TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0")

    result = adapter.parse_and_register_trispector_upload(upload, take_id="take_trispector_002")

    assert result["processing"]["status"] == "processing_completed"
    assert (settings.processed_dir / "take_trispector_002" / "DONE").is_file()


def test_trispector_take_thumbnail_prefers_height_preview(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    upload = tmp_path / "upload_thumb.png"
    _build_trispector_raw_image(upload)
    TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0").parse_and_register_trispector_upload(upload, take_id="take_thumb_25d")
    summary = get_take_summary(settings, "take_thumb_25d")
    assert summary.thumbnail_path == "heightmap_preview.png"


def test_trispector_adapter_reuses_existing_parser_function(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    upload = tmp_path / "upload_parser_reuse.png"
    _build_trispector_raw_image(upload)
    called = {"count": 0}

    from vision_3d_acquisition.acquisition import trispector_25d as module

    original = module.parse_trispector_2_5d_image

    def _wrapped(input_file: Path, output_dir: Path):
        called["count"] += 1
        return original(input_file, output_dir)

    monkeypatch.setattr(module, "parse_trispector_2_5d_image", _wrapped)
    TriSpectorFtpAcquisitionAdapter(settings, source_id="trispector_ftp_0").parse_and_register_trispector_upload(
        upload, take_id="take_trispector_parser_reuse"
    )
    assert called["count"] == 1
