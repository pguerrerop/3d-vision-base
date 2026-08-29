from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from vision_3d_acquisition.api.main import CreateFusionRunRequest, create_fusion_run
from vision_3d_acquisition.api.settings import ApiSettings


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


def _candidate_2d(object_id: str, *, class_hint: str | None = None) -> dict:
    return {
        "id": object_id,
        "object_id": object_id,
        "local_object_index": 1,
        "source_modality": "rgb",
        "bbox_px": [90, 90, 20, 20],
        "centroid_px": [100, 100],
        "geometry": {},
        "appearance": {},
        "measurements": {},
        "classification_hints": {"class_label": class_hint} if class_hint else {},
        "confidence": 0.9 if class_hint else 0.5,
        "diagnostics": {},
    }


def _candidate_25d(
    object_id: str,
    *,
    roundness: float = 0.9,
    deformation: float = 0.0,
    flatness: float = 0.5,
    max_height_mm: float = 10.0,
) -> dict:
    return {
        "id": object_id,
        "object_id": object_id,
        "local_object_index": 1,
        "source_modality": "derived_25d",
        "bbox_px": [88, 88, 24, 24],
        "centroid_px": [100, 100],
        "centroid_world": [1, 2, 3],
        "geometry": {},
        "appearance": {},
        "measurements": {"height_max_mm": max_height_mm},
        "classification_hints": {"deformation_hint": deformation, "roundness_hint": roundness, "flatness_hint": flatness},
        "confidence": 0.8,
        "diagnostics": {},
    }


def _write_result(data_dir: Path, take_id: str, *, family: str, candidates: list[dict]) -> Path:
    out = data_dir / "processed" / take_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success",
        "input_modalities": ["rgb"] if family == "2d" else ["heightmap"],
        "processing_pipeline": {"id": f"pipe_{family}", "pipeline_family": family},
        "object_candidates": candidates,
        "objects": [],
        "artifacts": [],
    }
    path = out / "result.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "DONE").touch()
    return path


def _ready_group(data_dir: Path, group_id: str, rgb_candidates: list[dict], hm_candidates: list[dict]) -> tuple[Path, Path]:
    _write_group(data_dir, group_id)
    _write_take(data_dir, "take_rgb_1", ["rgb"], group_id, "usb_camera_0")
    _write_take(data_dir, "take_hm_1", ["heightmap"], group_id, "trispector_ftp_0")
    rgb_path = _write_result(data_dir, "take_rgb_1", family="2d", candidates=rgb_candidates)
    hm_path = _write_result(data_dir, "take_hm_1", family="25d", candidates=hm_candidates)
    return rgb_path, hm_path


def test_not_ready_rejects_unless_force(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _write_group(settings.data_dir, "ag_not_ready")
    _write_take(settings.data_dir, "take_rgb_only", ["rgb"], "ag_not_ready", "usb_camera_0")
    _write_result(settings.data_dir, "take_rgb_only", family="2d", candidates=[_candidate_2d("rgb_1")])
    with pytest.raises(HTTPException):
        create_fusion_run("ag_not_ready", CreateFusionRunRequest(force=False), settings)
    forced = create_fusion_run("ag_not_ready", CreateFusionRunRequest(force=True), settings)
    assert forced["ok"] is True


def test_ready_creates_persisted_fusion_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(settings.data_dir, "ag_ready", [_candidate_2d("rgb_1", class_hint="Bola buena")], [_candidate_25d("hm_1")])
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
        [_candidate_2d("rgb_1", class_hint="Bola buena")],
        [_candidate_25d("hm_1", roundness=0.92, deformation=0.1)],
    )
    payload = create_fusion_run("ag_good", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class"] == "Bola buena"


def test_high_deformation_classifies_bola_deformada(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(
        settings.data_dir,
        "ag_deform",
        [_candidate_2d("rgb_1")],
        [_candidate_25d("hm_1", deformation=0.9)],
    )
    payload = create_fusion_run("ag_deform", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class_group"] == "Scrap de Bola"
    assert first["final_class"] == "Scrap de Bola / Bola deformada"


def test_elongated_or_flat_classifies_chatarra(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _ready_group(
        settings.data_dir,
        "ag_scrap",
        [_candidate_2d("rgb_1")],
        [_candidate_25d("hm_1", flatness=0.95, max_height_mm=3.0)],
    )
    payload = create_fusion_run("ag_scrap", CreateFusionRunRequest(), settings)
    first = payload["result"]["final_objects"][0]
    assert first["final_class_group"] == "Chatarra"


def test_warnings_preserved_and_source_outputs_unchanged(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    rgb_result_path, hm_result_path = _ready_group(
        settings.data_dir,
        "ag_warn",
        [_candidate_2d("rgb_1")],
        [_candidate_25d("hm_1")],
    )
    rgb_before = rgb_result_path.read_text(encoding="utf-8")
    hm_before = hm_result_path.read_text(encoding="utf-8")
    payload = create_fusion_run("ag_warn", CreateFusionRunRequest(rules={"unknown_knob": 1}), settings)
    assert payload["ok"] is True
    assert isinstance(payload["result"]["warnings"], list)
    assert rgb_result_path.read_text(encoding="utf-8") == rgb_before
    assert hm_result_path.read_text(encoding="utf-8") == hm_before
