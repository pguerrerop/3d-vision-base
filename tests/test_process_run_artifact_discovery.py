from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_3d_acquisition.api.filesystem import get_take_detail, safe_take_file
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.processes.service import ProcessService


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


def synthetic_rgb() -> np.ndarray:
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    image[:] = (25, 25, 25)
    cv2.circle(image, (160, 110), 48, (200, 200, 200), -1)
    return image


def test_take_detail_exposes_process_run_artifacts(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB POC",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    take_id = "take_rgb_001"
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), synthetic_rgb())

    executed = service.execute(instance["id"], image_path=image_path)
    run_id = executed["run"]["run_id"]
    detail = get_take_detail(settings, take_id)
    assert detail is not None
    assert detail.result is not None
    artifacts = detail.result.get("artifacts") or []
    by_id = {item.get("artifact_id"): item for item in artifacts if isinstance(item, dict)}

    for artifact_id in ("threshold_mask", "cleaned_mask", "overlay_image", "morphology_metrics", "morphology_debug_json"):
        assert artifact_id in by_id
        meta = by_id[artifact_id].get("metadata") or {}
        assert meta.get("run_id") == run_id
        assert meta.get("pipeline_instance_id") == instance["id"]
    assert by_id["threshold_mask"]["stage_id"] == "segmentation"
    assert by_id["cleaned_mask"]["stage_id"] == "segmentation"
    assert by_id["overlay_image"]["stage_id"] == "segmentation"
    object_candidates = detail.result.get("object_candidates")
    assert isinstance(object_candidates, list)
    assert len(object_candidates) >= 1
    first_candidate = object_candidates[0]
    assert first_candidate.get("source_modality") == "rgb"
    assert "geometry" in first_candidate


def test_safe_take_file_allows_process_run_nested_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB POC",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    take_id = "take_rgb_002"
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), synthetic_rgb())

    service.execute(instance["id"], image_path=image_path)
    detail = get_take_detail(settings, take_id)
    assert detail is not None and detail.result is not None
    artifacts = [item for item in (detail.result.get("artifacts") or []) if isinstance(item, dict)]
    threshold = next(item for item in artifacts if item.get("artifact_id") == "threshold_mask")
    path = str(threshold.get("path") or "")
    assert "processes/runs/" in path
    resolved = safe_take_file(settings, take_id, path)
    assert resolved is not None
    assert resolved.is_file()
