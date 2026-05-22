from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.main import acquisition_group_fusion_inputs, acquisition_group_fusion_preview
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.fusion.preparation import build_fusion_preview, resolve_fusion_inputs


def _write_take(data_dir: Path, take_id: str, source: str, modalities: list[str], group_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    if "rgb" in modalities:
        files["rgb"] = "rgb.png"
        (take_dir / "rgb.png").write_bytes(b"rgb")
    if "heightmap" in modalities:
        files["heightmap"] = "heightmap.npz"
        (take_dir / "heightmap.npz").write_bytes(b"hm")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": source,
                "created_at": f"2026-05-21T10:00:0{take_id[-1]}Z",
                "modalities": modalities,
                "acquisition_group_id": group_id,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def _write_process_entries(data_dir: Path, entries: list[dict]) -> None:
    path = data_dir / "processes" / "index" / "runs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def _write_result(data_dir: Path, take_id: str, family: str, *, objects: list[dict] | None = None) -> None:
    out = data_dir / "processed" / take_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success",
        "processing_pipeline": {"id": "pipe", "pipeline_family": family},
        "objects": objects if objects is not None else [],
        "artifacts": [{"artifact_id": "overlay", "path": f"{take_id}/overlay.png"}],
    }
    (out / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (out / "DONE").touch()


def _setup_group(data_dir: Path, group_id: str) -> None:
    groups = AcquisitionGroupService(data_dir)
    groups.groups_dir.mkdir(parents=True, exist_ok=True)
    (groups.groups_dir / f"{group_id}.json").write_text(
        json.dumps({"id": group_id, "status": "open", "started_at": "2026-05-21T10:00:00Z", "metadata": {}}, indent=2),
        encoding="utf-8",
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


def test_only_rgb_exists_waiting_for_heightmap(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_group(data_dir, "ag_1")
    _write_take(data_dir, "take1", "usb_camera_0", ["rgb"], "ag_1")
    _write_result(data_dir, "take1", "2d")
    _write_process_entries(data_dir, [{"take_id": "take1", "pipeline_family": "2d", "status": "success", "run_id": "r1", "created_at": "2026-05-21T10:10:00Z"}])
    bundle = resolve_fusion_inputs(data_dir, "ag_1")
    assert bundle.readiness_status == "waiting_for_heightmap"


def test_only_heightmap_exists_waiting_for_rgb(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_group(data_dir, "ag_2")
    _write_take(data_dir, "take2", "trispector_ftp_0", ["heightmap"], "ag_2")
    _write_result(data_dir, "take2", "25d")
    bundle = resolve_fusion_inputs(data_dir, "ag_2")
    assert bundle.readiness_status == "waiting_for_rgb"


def test_both_takes_exist_but_no_runs_waiting_for_processing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_group(data_dir, "ag_3")
    _write_take(data_dir, "take3", "usb_camera_0", ["rgb"], "ag_3")
    _write_take(data_dir, "take4", "trispector_ftp_0", ["heightmap"], "ag_3")
    _write_process_entries(data_dir, [])
    bundle = resolve_fusion_inputs(data_dir, "ag_3")
    assert bundle.readiness_status == "waiting_for_processing"
    assert "rgb_processed_run" in bundle.missing_inputs
    assert "heightmap_processed_run" in bundle.missing_inputs


def test_both_successful_runs_ready_for_fusion(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_group(data_dir, "ag_4")
    _write_take(data_dir, "take5", "usb_camera_0", ["rgb"], "ag_4")
    _write_take(data_dir, "take6", "trispector_ftp_0", ["heightmap"], "ag_4")
    _write_result(data_dir, "take5", "2d", objects=[{"object_id": "rgb_1"}])
    _write_result(data_dir, "take6", "25d", objects=[{"object_id": "hm_1"}])
    _write_process_entries(
        data_dir,
        [
            {"take_id": "take5", "pipeline_family": "2d", "status": "success", "run_id": "rgb_run", "created_at": "2026-05-21T10:11:00Z"},
            {"take_id": "take6", "pipeline_family": "25d", "status": "success", "run_id": "hm_run", "created_at": "2026-05-21T10:12:00Z"},
        ],
    )
    bundle = resolve_fusion_inputs(data_dir, "ag_4")
    assert bundle.readiness_status == "ready_for_fusion"
    assert bundle.rgb_run_id == "rgb_run"
    assert bundle.heightmap_run_id == "hm_run"


def test_multiple_candidates_generate_warning(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_group(data_dir, "ag_5")
    _write_take(data_dir, "take7", "usb_camera_0", ["rgb"], "ag_5")
    _write_take(data_dir, "take8", "usb_camera_0", ["rgb"], "ag_5")
    _write_take(data_dir, "take9", "trispector_ftp_0", ["heightmap"], "ag_5")
    _write_result(data_dir, "take7", "2d")
    _write_result(data_dir, "take8", "2d")
    _write_result(data_dir, "take9", "25d")
    _write_process_entries(
        data_dir,
        [
            {"take_id": "take7", "pipeline_family": "2d", "status": "success", "run_id": "r7", "created_at": "2026-05-21T10:10:00Z"},
            {"take_id": "take8", "pipeline_family": "2d", "status": "success", "run_id": "r8", "created_at": "2026-05-21T10:20:00Z"},
            {"take_id": "take9", "pipeline_family": "25d", "status": "success", "run_id": "r9", "created_at": "2026-05-21T10:21:00Z"},
        ],
    )
    bundle = resolve_fusion_inputs(data_dir, "ag_5")
    assert bundle.readiness_status == "ready_for_fusion"
    assert any("Multiple RGB candidates" in warning for warning in bundle.warnings)
    assert bundle.rgb_run_id == "r8"


def test_object_index_fallback_and_missing_fields_tolerant(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_group(data_dir, "ag_6")
    _write_take(data_dir, "take10", "usb_camera_0", ["rgb"], "ag_6")
    _write_take(data_dir, "take11", "trispector_ftp_0", ["heightmap"], "ag_6")
    _write_result(data_dir, "take10", "2d", objects=[{"object_id": "rgb_1", "class_hint": "Bola buena"}])
    _write_result(data_dir, "take11", "25d", objects=[{"object_id": "hm_1", "height_metrics": {"max_height_mm": 22.1}}])
    _write_process_entries(
        data_dir,
        [
            {"take_id": "take10", "pipeline_family": "2d", "status": "success", "run_id": "r10", "created_at": "2026-05-21T10:22:00Z"},
            {"take_id": "take11", "pipeline_family": "25d", "status": "success", "run_id": "r11", "created_at": "2026-05-21T10:23:00Z"},
        ],
    )
    preview = build_fusion_preview(data_dir, "ag_6")
    assert preview.readiness_status == "ready_for_fusion"
    assert preview.object_candidates
    assert preview.object_candidates[0].matching_method == "object_index_fallback"
    assert isinstance(preview.warnings, list)


def test_fusion_api_endpoints_return_readiness_payload(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = _settings(data_dir)
    _setup_group(data_dir, "ag_7")
    _write_take(data_dir, "take12", "usb_camera_0", ["rgb"], "ag_7")
    payload_inputs = acquisition_group_fusion_inputs("ag_7", settings)
    payload_preview = acquisition_group_fusion_preview("ag_7", settings)
    assert payload_inputs["acquisition_group_id"] == "ag_7"
    assert payload_inputs["readiness_status"] == "waiting_for_heightmap"
    assert payload_preview["acquisition_group_id"] == "ag_7"
    assert "object_candidates" in payload_preview
