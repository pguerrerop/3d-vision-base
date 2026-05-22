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


def synthetic_rgb_noise_and_ball() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    noise = rng.integers(0, 40, size=image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    cv2.circle(image, (160, 120), 48, (205, 205, 205), -1)
    for x, y in [(20, 20), (300, 20), (20, 220), (300, 220)]:
        cv2.circle(image, (x, y), 2, (190, 190, 190), -1)
    return image


def test_segmentation_artifacts_and_debug_payload(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB segmentation semantics",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    configured = instance["configured_steps"]
    for step in configured:
        if step["step_id"] == "threshold":
            step["params"].update(
                {
                    "mode": "otsu",
                    "invert": False,
                    "blur_kernel": 5,
                    "roi_enabled": True,
                    "roi_x": 40,
                    "roi_y": 40,
                    "roi_width": 220,
                    "roi_height": 160,
                }
            )
        if step["step_id"] == "morphology":
            step["params"].update(
                {
                    "operation": "open_close",
                    "open_kernel": 3,
                    "close_kernel": 5,
                    "cleanup_min_area": 60,
                    "overlay_alpha": 0.35,
                }
            )
    service.update_instance(instance["id"], configured_steps=configured)

    take_dir = settings.incoming_dir / "take_seg_semantics"
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), synthetic_rgb_noise_and_ball())

    run = service.execute(instance["id"], image_path=image_path)["run"]
    by_id = {item["artifact_id"]: item for item in run["artifacts"]}
    assert "threshold_mask" in by_id
    assert "cleaned_mask" in by_id
    assert "overlay_image" in by_id
    assert "morphology_debug_json" in by_id
    assert by_id["overlay_image"]["metadata"]["overlay_type"] == "segmentation_mask"

    threshold_path = settings.data_dir / str(by_id["threshold_mask"]["path"])
    cleaned_path = settings.data_dir / str(by_id["cleaned_mask"]["path"])
    overlay_path = settings.data_dir / str(by_id["overlay_image"]["path"])
    src = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    threshold = cv2.imread(str(threshold_path), cv2.IMREAD_GRAYSCALE)
    cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert src is not None and threshold is not None and cleaned is not None and overlay is not None

    roi = by_id["morphology_debug_json"]["metadata"]["roi"]
    rx, ry, rw, rh = int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])
    outside = np.ones_like(cleaned, dtype=bool)
    outside[ry:ry + rh, rx:rx + rw] = False
    assert int(np.count_nonzero(cleaned[outside])) == 0

    background = cleaned == 0
    assert np.array_equal(overlay[background], src[background])

    payload = by_id["morphology_debug_json"]["metadata"]
    assert payload["stage"] == "segmentation"
    assert "threshold" in payload and "morphology" in payload and "roi" in payload and "artifacts" in payload
    assert isinstance(payload.get("effective_params"), dict)


def test_polygon_roi_masks_outside_polygon_and_metrics_are_coherent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB segmentation polygon roi",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    configured = instance["configured_steps"]
    polygon = [[80, 60], [260, 70], [250, 180], [90, 170]]
    for step in configured:
        if step["step_id"] == "threshold":
            step["params"].update(
                {
                    "mode": "otsu",
                    "invert": False,
                    "blur_kernel": 5,
                    "roi_enabled": True,
                    "roi_type": "polygon",
                    "roi_polygon_points": polygon,
                }
            )
        if step["step_id"] == "morphology":
            step["params"].update(
                {
                    "operation": "open_close",
                    "open_kernel": 3,
                    "close_kernel": 5,
                    "cleanup_min_area": 40,
                }
            )
    service.update_instance(instance["id"], configured_steps=configured)

    take_dir = settings.incoming_dir / "take_seg_poly"
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), synthetic_rgb_noise_and_ball())

    run = service.execute(instance["id"], image_path=image_path)["run"]
    by_id = {item["artifact_id"]: item for item in run["artifacts"]}
    cleaned_path = settings.data_dir / str(by_id["cleaned_mask"]["path"])
    cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_GRAYSCALE)
    assert cleaned is not None

    roi = by_id["morphology_debug_json"]["metadata"]["roi"]
    assert roi["type"] == "polygon"
    assert len(roi["polygon_points"]) >= 3

    roi_mask = np.zeros_like(cleaned, dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.asarray(roi["polygon_points"], dtype=np.int32)], 255)
    outside = roi_mask == 0
    assert int(np.count_nonzero(cleaned[outside])) == 0

    metrics = by_id["morphology_metrics"]["metadata"]
    assert int(metrics["components_after"]) >= 0
    assert float(metrics["cleaned_foreground_coverage"]) >= 0.0


def test_cleanup_filters_reject_small_and_border_components(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB cleanup filters",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    configured = instance["configured_steps"]
    for step in configured:
        if step["step_id"] == "threshold":
            step["params"].update({"mode": "fixed", "value": 90, "invert": False, "roi_enabled": False})
        if step["step_id"] == "morphology":
            step["params"].update(
                {
                    "operation": "none",
                    "cleanup_min_area": 100,
                    "cleanup_border_reject": True,
                }
            )
    service.update_instance(instance["id"], configured_steps=configured)
    image = np.zeros((180, 260, 3), dtype=np.uint8)
    image[:] = (20, 20, 20)
    cv2.circle(image, (120, 90), 30, (230, 230, 230), -1)
    cv2.circle(image, (10, 10), 4, (230, 230, 230), -1)
    take_dir = settings.incoming_dir / "take_cleanup_filters"
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), image)
    run = service.execute(instance["id"], image_path=image_path)["run"]
    by_id = {item["artifact_id"]: item for item in run["artifacts"]}
    metrics = by_id["morphology_metrics"]["metadata"]
    assert int(metrics["rejected_small"]) >= 1 or int(metrics["rejected_border"]) >= 1
    assert int(metrics["components_after_cleanup"]) >= 1
    blob_metrics = by_id["blob_metrics"]["metadata"]
    assert isinstance(blob_metrics.get("params"), dict)
    assert "rejected_reason_counts" in blob_metrics
