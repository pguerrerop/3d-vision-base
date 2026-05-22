from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.api.main import (
    FuseAcquisitionGroupRequest,
    fuse_acquisition_group,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.fusion.service import FusionService
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


def _write_take(settings: ApiSettings, take_id: str, *, group_id: str, with_rgb: bool, with_heightmap: bool) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    modalities: list[str] = []
    if with_rgb:
        files["rgb"] = "rgb.png"
        (take_dir / "rgb.png").write_bytes(b"x")
        modalities.append("rgb")
    if with_heightmap:
        files["heightmap"] = "heightmap.npz"
        (take_dir / "heightmap.npz").write_bytes(b"x")
        modalities.append("heightmap")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "created_at": "2026-05-21T12:00:00Z",
                "acquisition_group_id": group_id,
                "files": files,
                "modalities": modalities,
            }
        ),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def _write_result(settings: ApiSettings, take_id: str, *, family: str, candidates: list[dict]) -> None:
    out = settings.processed_dir / take_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "take_id": take_id,
        "processed_at": datetime.now(UTC).isoformat(),
        "processing_mode": "real",
        "processing_engine": "native",
        "input_modalities": ["rgb"] if family == "2d" else ["heightmap"],
        "algorithm_stage": "classification",
        "status": "ok",
        "summary": {"object_count": len(candidates), "ball_count": 0, "non_ball_count": 0, "decision": "review", "confidence": None},
        "objects": [],
        "object_candidates": candidates,
        "files": {},
        "timing_ms": {"load": 0, "segmentation": 0, "classification": 0, "total": 0},
        "error": None,
        "artifacts": [],
        "processing_pipeline": {"id": "mining_steel_ball_classification_2d" if family == "2d" else "mining_steel_ball_classification_25d", "pipeline_family": family},
    }
    (out / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (out / "DONE").touch()


def _candidate_2d(cid: str, x: float, y: float) -> dict:
    return {
        "id": cid,
        "local_object_index": 1,
        "source_modality": "rgb",
        "source_take_id": "t_rgb",
        "source_pipeline_run_id": "run2d",
        "bbox_px": [x - 10, y - 10, 20, 20],
        "centroid_px": [x, y],
        "geometry": {},
        "appearance": {},
        "measurements": {},
        "classification_hints": {"class_label": "Bola buena"},
        "confidence": 0.9,
        "diagnostics": {},
    }


def _candidate_25d(cid: str, x: float, y: float, deformation: float = 0.0) -> dict:
    return {
        "id": cid,
        "local_object_index": 1,
        "source_modality": "derived_25d",
        "source_take_id": "t_hm",
        "source_pipeline_run_id": "run25d",
        "bbox_px": [x - 12, y - 12, 24, 24],
        "centroid_px": [x, y],
        "centroid_world": [1, 2, 3],
        "geometry": {},
        "appearance": {},
        "measurements": {"height_max_mm": 10.0},
        "classification_hints": {"deformation_hint": deformation, "roundness_hint": 0.9, "flatness_hint": 0.5},
        "confidence": 0.8,
        "diagnostics": {},
    }


def test_loads_candidates_from_group_and_matches_by_centroid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g1")
    _write_take(settings, "take_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
    _write_take(settings, "take_hm", group_id=group["id"], with_rgb=False, with_heightmap=True)
    _write_result(settings, "take_rgb", family="2d", candidates=[_candidate_2d("c2d_1", 100, 120)])
    _write_result(settings, "take_hm", family="25d", candidates=[_candidate_25d("c25d_1", 103, 119)])

    payload = FusionService(settings).run_fusion(acquisition_group_id=group["id"], recipe_version_id=None)
    assert len(payload["matched_objects"]) == 1
    assert len(payload["final_objects"]) >= 1


def test_unmatched_candidates_preserved(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g2")
    _write_take(settings, "take_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
    _write_take(settings, "take_hm", group_id=group["id"], with_rgb=False, with_heightmap=True)
    _write_result(settings, "take_rgb", family="2d", candidates=[_candidate_2d("c2d_1", 10, 10)])
    _write_result(settings, "take_hm", family="25d", candidates=[_candidate_25d("c25d_1", 300, 300)])

    payload = FusionService(settings).run_fusion(acquisition_group_id=group["id"], recipe_version_id=None)
    assert len(payload["matched_objects"]) == 0
    assert len(payload["unmatched_2d_candidates"]) == 1
    assert len(payload["unmatched_25d_candidates"]) == 1


def test_classification_rule_emits_bola_deformada(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g3")
    _write_take(settings, "take_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
    _write_take(settings, "take_hm", group_id=group["id"], with_rgb=False, with_heightmap=True)
    _write_result(settings, "take_rgb", family="2d", candidates=[_candidate_2d("c2d_1", 100, 120)])
    _write_result(settings, "take_hm", family="25d", candidates=[_candidate_25d("c25d_1", 100, 120, deformation=0.9)])

    payload = FusionService(settings).run_fusion(acquisition_group_id=group["id"], recipe_version_id=None)
    classes = [item.get("final_class") for item in payload["final_objects"]]
    assert "Scrap de Bola / Bola deformada" in classes


def test_endpoint_with_explicit_recipe_version_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g4")
    _write_take(settings, "take_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
    _write_take(settings, "take_hm", group_id=group["id"], with_rgb=False, with_heightmap=True)
    _write_result(settings, "take_rgb", family="2d", candidates=[_candidate_2d("c2d_1", 100, 120)])
    _write_result(settings, "take_hm", family="25d", candidates=[_candidate_25d("c25d_1", 100, 120)])

    payload = fuse_acquisition_group(group["id"], FuseAcquisitionGroupRequest(recipe_version_id="recipe_fusion_1"), settings)
    assert payload["diagnostics"]["recipe_version_id"] == "recipe_fusion_1"


def test_endpoint_works_with_process_binding_purpose_fusion(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g5")
    _write_take(settings, "take_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
    _write_take(settings, "take_hm", group_id=group["id"], with_rgb=False, with_heightmap=True)
    _write_result(settings, "take_rgb", family="2d", candidates=[_candidate_2d("c2d_1", 100, 120)])
    _write_result(settings, "take_hm", family="25d", candidates=[_candidate_25d("c25d_1", 100, 120)])

    recipes_dir = settings.data_dir / "processes" / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    recipe_id = "recipe_fusion_bind"
    (recipes_dir / f"{recipe_id}.json").write_text(
        json.dumps(
            {
                "id": recipe_id,
                "pipeline_instance_id": "fusion",
                "source_run_id": None,
                "type": "production",
                "notes": None,
                "created_at": datetime.now(UTC).isoformat(),
                "template_id": "fusion",
                "pipeline_snapshot": {"pipeline_id": "mining_steel_ball_fusion"},
            }
        ),
        encoding="utf-8",
    )
    ProcessBindingService(settings).create_binding(
        name="fusion binding",
        source_id=group["id"],
        modality="acquisition_group",
        purpose="fusion",
        pipeline_id="mining_steel_ball_fusion",
        active_recipe_version_id=recipe_id,
        calibration_profile_id=None,
        enabled=True,
    )

    payload = fuse_acquisition_group(group["id"], FuseAcquisitionGroupRequest(recipe_version_id=None), settings)
    assert payload["diagnostics"]["recipe_version_id"] == recipe_id


def test_missing_candidates_returns_diagnostic_not_crash(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="g6")
    _write_take(settings, "take_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
    _write_result(settings, "take_rgb", family="2d", candidates=[_candidate_2d("c2d_1", 100, 120)])

    payload = FusionService(settings).run_fusion(acquisition_group_id=group["id"], recipe_version_id=None)
    assert payload["diagnostics"]["message"] == "missing candidates"
    assert isinstance(payload["final_objects"], list)
