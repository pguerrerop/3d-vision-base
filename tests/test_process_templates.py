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


def _synthetic_rgb_image() -> np.ndarray:
    image = np.zeros((260, 360, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    cv2.circle(image, (110, 135), 42, (210, 210, 210), -1)
    cv2.circle(image, (240, 140), 30, (185, 185, 185), -1)
    cv2.rectangle(image, (292, 50), (336, 102), (170, 170, 170), -1)
    return image


def test_mining_2d_template_pipeline_executes_end_to_end(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)

    templates = service.list_templates()
    mining = next(item for item in templates if item["id"] == "mining_steel_ball_classification_2d_reflectance_mvp")
    assert "RGB/2D" in mining["name"]
    assert mining["ui_metadata"]["default_input_type"] == "rgb_image"

    instance = service.create_instance(
        name="POC Mining RGB",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )

    take_dir = settings.data_dir / "incoming" / "take_rgb"
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), _synthetic_rgb_image())

    executed = service.execute(instance["id"], image_path=image_path)
    run = executed["run"]

    assert run["status"] in {"success", "warning"}
    ids = {artifact["artifact_id"] for artifact in run["artifacts"]}
    assert "grayscale_image" in ids
    assert "threshold_mask" in ids
    assert "cleaned_mask" in ids
    assert "overlay_image" in ids
    assert "morphology_metrics" in ids
    assert "morphology_debug_json" in ids
    assert "blob_debug_overlay" in ids
    assert "blob_contours" in ids
    assert "blob_metrics" in ids
    assert "ellipse_overlay" in ids
    assert "ellipse_metrics" in ids
    assert "ellipse_summary" in ids
    assert "measurement_table_artifact" in ids
    assert "classification_result_artifact" in ids
    assert any(item["kind"] == "overlay" and item["coordinate_space"] == "image_pixel" for item in run["artifacts"])
    assert len(run["measurements"]) >= 1
    by_id = {artifact["artifact_id"]: artifact for artifact in run["artifacts"]}
    ellipse_rows = ((by_id["ellipse_metrics"].get("metadata") or {}).get("entries") or [])
    assert isinstance(ellipse_rows, list)
    if ellipse_rows:
        row = ellipse_rows[0]
        assert "equivalent_diameter" in row
        assert "eccentricity" in row
        assert "fit_rmse" in row
        assert "valid_fit" in row
    ellipse_summary = by_id["ellipse_summary"].get("metadata") or {}
    assert int(ellipse_summary.get("candidate_count", 0)) >= 1
    assert int(ellipse_summary.get("fitted_count", 0)) >= 1


def test_mining_rgb_pipeline_supports_parameter_override_and_rerun(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    service = ProcessService(settings)
    instance = service.create_instance(
        name="POC Mining RGB",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )

    take_dir = settings.data_dir / "incoming" / "take_rgb"
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    cv2.imwrite(str(image_path), _synthetic_rgb_image())

    first = service.execute(instance["id"], image_path=image_path)["run"]

    configured_steps = []
    for step in service.list_instances()[0]["configured_steps"]:
        if step["step_id"] == "threshold":
            step = {**step, "params": {**step["params"], "value": 180}}
        configured_steps.append(step)
    service.update_instance(instance["id"], configured_steps=configured_steps)
    second = service.execute(instance["id"], image_path=image_path)["run"]

    assert second["run_id"] != first["run_id"]
    assert second["parameters"]["threshold"]["value"] == 180
