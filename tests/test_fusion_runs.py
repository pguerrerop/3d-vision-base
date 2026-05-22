from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from vision_3d_acquisition.api.main import CreateFusionRunRequest, create_fusion_run
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.fusion.service import FusionService


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


def _write_group(data_dir: Path, group_id: str) -> None:
    group_dir = data_dir / "acquisition_groups" / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / f"{group_id}.json").write_text(
        json.dumps({"id": group_id, "status": "open", "started_at": "2026-05-21T12:00:00Z", "metadata": {}}, indent=2),
        encoding="utf-8",
    )


def _write_take(data_dir: Path, take_id: str, modalities: list[str], group_id: str, source: str) -> None:
    td = data_dir / "incoming" / take_id
    td.mkdir(parents=True, exist_ok=True)
    files = {}
    if "rgb" in modalities:
        files["rgb"] = "rgb.png"
        (td / "rgb.png").write_bytes(b"rgb")
    if "heightmap" in modalities:
        files["heightmap"] = "heightmap.npz"
        (td / "heightmap.npz").write_bytes(b"hm")
    (td / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": source,
                "created_at": f"2026-05-21T12:00:0{take_id[-1]}Z",
                "modalities": modalities,
                "acquisition_group_id": group_id,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    (td / "READY").touch()


def _write_result(data_dir: Path, take_id: str, family: str, objects: list[dict]) -> Path:
    out = data_dir / "processed" / take_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success",
        "processing_pipeline": {"id": f"pipe_{family}", "pipeline_family": family},
        "objects": objects,
        "artifacts": [],
    }
    path = out / "result.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "DONE").touch()
    return path


def _write_runs_index(data_dir: Path, entries: list[dict]) -> None:
    path = data_dir / "processes" / "index" / "runs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def _ready_group(data_dir: Path, group_id: str, rgb_objects: list[dict], hm_objects: list[dict]) -> tuple[Path, Path]:
    _write_group(data_dir, group_id)
    _write_take(data_dir, "take_rgb_1", ["rgb"], group_id, "usb_camera_0")
    _write_take(data_dir, "take_hm_1", ["heightmap"], group_id, "trispector_ftp_0")
    rgb_path = _write_result(data_dir, "take_rgb_1", "2d", rgb_objects)
    hm_path = _write_result(data_dir, "take_hm_1", "25d", hm_objects)
    _write_runs_index(
        data_dir,
        [
            {"take_id": "take_rgb_1", "pipeline_family": "2d", "status": "success", "run_id": "rgb_run_1", "created_at": "2026-05-21T12:00:10Z"},
            {"take_id": "take_hm_1", "pipeline_family": "25d", "status": "success", "run_id": "hm_run_1", "created_at": "2026-05-21T12:00:11Z"},
        ],
    )
    return rgb_path, hm_path


def test_not_ready_rejects_unless_force(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _write_group(settings.data_dir, "ag_not_ready")
    _write_take(settings.data_dir, "take_rgb_only", ["rgb"], "ag_not_ready", "usb_camera_0")
    _write_result(settings.data_dir, "take_rgb_only", "2d", [{"object_id": "rgb_1"}])
    _write_runs_index(settings.data_dir, [{"take_id": "take_rgb_only", "pipeline_family": "2d", "status": "success", "run_id": "r1", "created_at": "2026-05-21T12:10:00Z"}])
    with pytest.raises(HTTPException):
        create_fusion_run("ag_not_ready", CreateFusionRunRequest(force=False), settings)
    forced = create_fusion_run("ag_not_ready", CreateFusionRunRequest(force=True), settings)
    assert forced["ok"] is True


def test_ready_creates_persisted_fusion_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(settings.data_dir, "ag_ready", [{"object_id": "rgb_1", "class_hint": "Bola buena"}], [{"object_id": "hm_1", "footprint_geometry": {"roundness": 0.9}}])
    payload = create_fusion_run("ag_ready", CreateFusionRunRequest(), settings)
    assert payload["ok"] is True
    run_path = Path(payload["run"]["result_path"])
    assert run_path.is_file()
    run_dir = run_path.parent
    assert (run_dir / "fusion_run.json").is_file()
    assert (run_dir / "fusion_summary.json").is_file()
    assert (run_dir / "fusion_object_table.json").is_file()
    assert (run_dir / "fusion_debug_pairing.json").is_file()


def test_good_rgb_and_good_25d_classifies_bola_buena(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(
        settings.data_dir,
        "ag_good",
        [{"object_id": "rgb_1", "class_hint": "Bola buena"}],
        [{"object_id": "hm_1", "footprint_geometry": {"roundness": 0.92}, "height_metrics": {"deformation_score": 0.1}}],
    )
    payload = create_fusion_run("ag_good", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class"] == "Bola buena"


def test_high_deformation_classifies_bola_deformada(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(
        settings.data_dir,
        "ag_deform",
        [{"object_id": "rgb_1"}],
        [{"object_id": "hm_1", "height_metrics": {"deformation_score": 0.9}}],
    )
    payload = create_fusion_run("ag_deform", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class_group"] == "Scrap de Bola"
    assert first["final_class"] == "Bola deformada"


def test_rgb_chip_hint_classifies_bola_con_chip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(
        settings.data_dir,
        "ag_chip",
        [{"object_id": "rgb_1", "class_hint": "chip visible"}],
        [{"object_id": "hm_1", "height_metrics": {"deformation_score": 0.05}}],
    )
    payload = create_fusion_run("ag_chip", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class_group"] == "Scrap de Bola"
    assert first["final_class"] == "Bola con chip"


def test_elongated_or_flat_classifies_chatarra(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(
        settings.data_dir,
        "ag_scrap",
        [{"object_id": "rgb_1"}],
        [{"object_id": "hm_1", "footprint_geometry": {"major_axis_mm": 100.0, "minor_axis_mm": 20.0, "flatness": 0.95}, "height_metrics": {"max_height_mm": 5.0}}],
    )
    payload = create_fusion_run("ag_scrap", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class_group"] == "Chatarra"


def test_warnings_preserved_and_source_outputs_unchanged(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    rgb_result_path, hm_result_path = _ready_group(
        settings.data_dir,
        "ag_warn",
        [{"object_id": "rgb_1"}],
        [{"object_id": "hm_1"}],
    )
    rgb_before = rgb_result_path.read_text(encoding="utf-8")
    hm_before = hm_result_path.read_text(encoding="utf-8")
    payload = FusionService(settings).run_fusion(acquisition_group_id="ag_warn", force=False, recipe_version_id=None, rules={"unknown_knob": 1})
    assert payload["ok"] is True
    assert isinstance(payload["result"]["warnings"], list)
    assert rgb_result_path.read_text(encoding="utf-8") == rgb_before
    assert hm_result_path.read_text(encoding="utf-8") == hm_before
