from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.api.main import (
    CreateFusionRunRequest,
    create_fusion_run,
    get_operator_inspection_result,
    latest_operator_inspection_result,
    list_operator_inspection_results,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.fusion.published_service import publish_fusion_result
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
                "created_at": datetime.now(UTC).isoformat(),
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
        "processing_pipeline": {"id": f"pipeline_{family}", "pipeline_family": family},
    }
    (out / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (out / "DONE").touch()


def _candidate_2d(cid: str, x: float, y: float, *, class_label: str = "Bola buena", confidence: float = 0.95, with_overlay: bool = True) -> dict:
    payload = {
        "id": cid,
        "source_modality": "rgb",
        "source_take_id": "t_rgb",
        "bbox_px": [x - 10, y - 10, 20, 20],
        "centroid_px": [x, y],
        "geometry": {},
        "appearance": {},
        "measurements": {"diameter_mm": 22.1},
        "classification_hints": {"class_label": class_label},
        "confidence": confidence,
    }
    if with_overlay:
        payload["overlay_artifact_id"] = f"overlay_{cid}"
    return payload


def _candidate_25d(cid: str, x: float, y: float, *, with_overlay: bool = True) -> dict:
    payload = {
        "id": cid,
        "source_modality": "derived_25d",
        "source_take_id": "t_hm",
        "bbox_px": [x - 12, y - 12, 24, 24],
        "centroid_px": [x, y],
        "centroid_world": [1.0, 2.0, 3.0],
        "geometry": {},
        "appearance": {},
        "measurements": {"height_max_mm": 10.0},
        "classification_hints": {"deformation_hint": 0.1, "roundness_hint": 0.9, "flatness_hint": 0.4},
        "confidence": 0.9,
    }
    if with_overlay:
        payload["overlay_artifact_id"] = f"overlay_{cid}"
    return payload


def _build_group_with_candidates(
    settings: ApiSettings,
    *,
    station_id: str | None = None,
    session_id: str | None = None,
    rgb_candidates: list[dict] | None = None,
    hm_candidates: list[dict] | None = None,
) -> str:
    metadata = {"session_id": session_id} if session_id else {}
    group = AcquisitionGroupService(settings.data_dir).create_acquisition_group(name="group", station_id=station_id, metadata=metadata)
    if rgb_candidates is not None:
        _write_take(settings, f"{group['id']}_rgb", group_id=group["id"], with_rgb=True, with_heightmap=False)
        _write_result(settings, f"{group['id']}_rgb", family="2d", candidates=rgb_candidates)
    if hm_candidates is not None:
        _write_take(settings, f"{group['id']}_hm", group_id=group["id"], with_rgb=False, with_heightmap=True)
        _write_result(settings, f"{group['id']}_hm", family="25d", candidates=hm_candidates)
    return group["id"]


def _run_and_publish(settings: ApiSettings, group_id: str, *, force: bool = False) -> dict:
    fusion = FusionService(settings).run_fusion(acquisition_group_id=group_id, force=force)
    return publish_fusion_result(settings.data_dir, str(fusion["fusion_run_id"])).model_dump(mode="json")


def test_publish_bola_buena_maps_to_accept(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120, class_label="Bola buena", confidence=0.96)],
        hm_candidates=[_candidate_25d("c25d_1", 101, 121)],
    )
    published = _run_and_publish(settings, group_id)
    assert published["overall_decision"] == "accept"
    assert published["primary_class"] == "Bola buena"


def test_publish_scrap_de_bola_maps_to_reject(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120, class_label="Scrap de Bola", confidence=0.97)],
        hm_candidates=[_candidate_25d("c25d_1", 101, 121)],
    )
    published = _run_and_publish(settings, group_id)
    assert published["overall_decision"] == "reject"


def test_publish_chatarra_maps_to_reject(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120, class_label="Chatarra", confidence=0.97)],
        hm_candidates=[_candidate_25d("c25d_1", 101, 121)],
    )
    published = _run_and_publish(settings, group_id)
    assert published["overall_decision"] == "reject"


def test_low_confidence_or_warnings_map_to_review(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120, class_label="Bola buena", confidence=0.6)],
        hm_candidates=None,
    )
    published = _run_and_publish(settings, group_id, force=True)
    assert published["overall_decision"] == "review"
    assert published["status"] in {"incomplete", "failed", "complete"}


def test_missing_fusion_artifacts_still_publishes(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120, with_overlay=False)],
        hm_candidates=[_candidate_25d("c25d_1", 101, 121, with_overlay=False)],
    )
    published = _run_and_publish(settings, group_id)
    assert published["published_result_id"]
    assert isinstance(published["display_artifacts"], dict)
    assert "main_overlay" in published["display_artifacts"]


def test_index_latest_pointer_and_latest_by_station(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_a = _build_group_with_candidates(
        settings,
        station_id="station_a",
        session_id="session_a",
        rgb_candidates=[_candidate_2d("c2d_a", 100, 120)],
        hm_candidates=[_candidate_25d("c25d_a", 101, 121)],
    )
    group_b = _build_group_with_candidates(
        settings,
        station_id="station_b",
        session_id="session_b",
        rgb_candidates=[_candidate_2d("c2d_b", 100, 120)],
        hm_candidates=[_candidate_25d("c25d_b", 101, 121)],
    )

    published_a = _run_and_publish(settings, group_a)
    published_b = _run_and_publish(settings, group_b)
    index_path = settings.data_dir / "published" / "inspection_results" / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert index_payload["latest_global"] == published_b["published_result_id"]
    assert index_payload["latest_by_station"]["station_a"] == published_a["published_result_id"]
    assert index_payload["latest_by_station"]["station_b"] == published_b["published_result_id"]

    latest_station_a = latest_operator_inspection_result(station_id="station_a", settings=settings)
    assert latest_station_a["published_result_id"] == published_a["published_result_id"]


def test_operator_api_returns_published_shape_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        station_id="station_api",
        session_id="session_api",
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120)],
        hm_candidates=[_candidate_25d("c25d_1", 101, 121)],
    )
    published = _run_and_publish(settings, group_id)
    latest = latest_operator_inspection_result(station_id="station_api", settings=settings)
    fetched = get_operator_inspection_result(published["published_result_id"], settings=settings)
    listed = list_operator_inspection_results(station_id="station_api", session_id="session_api", limit=10, settings=settings)

    expected_keys = {
        "published_result_id",
        "acquisition_group_id",
        "fusion_run_id",
        "station_id",
        "session_id",
        "timestamp",
        "status",
        "overall_decision",
        "primary_class",
        "primary_class_group",
        "confidence",
        "class_counts",
        "objects",
        "warnings",
        "display_artifacts",
        "source_refs",
    }
    assert set(latest.keys()) == expected_keys
    assert set(fetched.keys()) == expected_keys
    assert listed["results"][0]["published_result_id"] == published["published_result_id"]
    assert "matched_objects" not in latest
    assert "unmatched_2d_candidates" not in latest
    assert "diagnostics" not in latest


def test_auto_publish_on_fusion_run_creates_published_result(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    group_id = _build_group_with_candidates(
        settings,
        station_id="station_auto",
        rgb_candidates=[_candidate_2d("c2d_1", 100, 120)],
        hm_candidates=[_candidate_25d("c25d_1", 101, 121)],
    )
    response = create_fusion_run(group_id, CreateFusionRunRequest(auto_publish=True), settings)
    published_result_id = response.get("published_result_id")
    assert isinstance(published_result_id, str) and published_result_id

    result_dir = settings.data_dir / "published" / "inspection_results" / published_result_id
    assert (result_dir / "published_result.json").is_file()
    assert (result_dir / "display_summary.json").is_file()
