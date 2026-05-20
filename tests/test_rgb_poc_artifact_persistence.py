from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

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


def synthetic_ball() -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.circle(img, (150, 130), 45, (210, 210, 210), -1)
    return img


def synthetic_oval() -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.ellipse(img, (160, 120), (84, 42), 20, 0, 360, (210, 210, 210), -1)
    return img


def test_persists_intermediate_image_artifacts_and_history_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(name="RGB", template_id="mining_steel_ball_classification_2d_reflectance_mvp", supported_input_type="rgb_image")

    take_dir = settings.data_dir / "incoming" / "take"
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), synthetic_ball())

    run = service.execute(instance["id"], image_path=image_path)["run"]

    needed = {
        "source_rgb_image",
        "grayscale_image",
        "normalized_grayscale_image",
        "threshold_mask",
        "cleaned_mask",
        "overlay_image",
        "rejected_components_overlay",
        "morphology_metrics",
        "morphology_debug_json",
        "blob_debug_overlay",
        "blob_labels",
        "blob_contours",
        "blob_metrics",
    }
    by_id = {item["artifact_id"]: item for item in run["artifacts"]}
    for artifact_id in needed:
        assert artifact_id in by_id
        path = by_id[artifact_id].get("path")
        assert isinstance(path, str) and path
        assert (settings.data_dir / path).is_file()
        metadata = by_id[artifact_id].get("metadata") or {}
        assert metadata.get("coordinate_space") == "image_pixel"
        assert metadata.get("step_id")
        assert metadata.get("algorithm_key")


def test_rerun_creates_distinct_artifact_paths_and_ball_vs_oval_behavior(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(name="RGB", template_id="mining_steel_ball_classification_2d_reflectance_mvp", supported_input_type="rgb_image")

    take_dir = settings.data_dir / "incoming" / "take"
    take_dir.mkdir(parents=True, exist_ok=True)

    ball_path = take_dir / "ball.png"
    cv2.imwrite(str(ball_path), synthetic_ball())
    first = service.execute(instance["id"], image_path=ball_path)["run"]
    assert int(first["summary"].get("balls", 0)) >= 1

    tuned_steps = []
    for step in service.list_instances()[0]["configured_steps"]:
        if step["step_id"] == "normalize_lighting":
            step = {**step, "params": {**step["params"], "method": "none"}}
        if step["step_id"] == "threshold":
            step = {**step, "params": {**step["params"], "value": 90, "mode": "binary"}}
        if step["step_id"] == "blob_detection":
            step = {**step, "params": {**step["params"], "min_circularity": 0.0}}
        tuned_steps.append(step)
    service.update_instance(instance["id"], configured_steps=tuned_steps)

    oval_path = take_dir / "oval.png"
    cv2.imwrite(str(oval_path), synthetic_oval())
    second = service.execute(instance["id"], image_path=oval_path)["run"]
    assert second["run_id"] != first["run_id"]
    assert int(second["summary"].get("total_objects", 0)) >= 1
    assert int(second["summary"].get("non_spherical_enough", 0)) >= 1 or int(second["summary"].get("non_balls", 0)) >= 1

    first_source = next(item for item in first["artifacts"] if item["artifact_id"] == "source_rgb_image")["path"]
    second_source = next(item for item in second["artifacts"] if item["artifact_id"] == "source_rgb_image")["path"]
    assert first_source != second_source

    overlays = [item for item in second["artifacts"] if item.get("kind") == "overlay"]
    assert overlays
    assert all(item.get("coordinate_space") == "image_pixel" for item in overlays)
