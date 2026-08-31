from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_3d_acquisition.api.main import (
    ActivateProcessBindingRecipeRequest,
    CreateProcessBindingRequest,
    ExecuteTakeRequest,
    create_process_binding,
    list_process_bindings,
    process_take_for_pipeline,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.processes.bindings import ProcessBindingService
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


def _rgb_take(settings: ApiSettings, take_id: str = "take_rgb_bindings") -> Path:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((140, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 70), 36, (220, 220, 220), -1)
    cv2.imwrite(str(take_dir / "rgb.png"), image)
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "usb_camera_0",
                "created_at": "2026-05-20T10:00:00Z",
                "files": {"rgb": "rgb.png"},
            }
        ),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()
    return take_dir


def _recipe(settings: ApiSettings) -> tuple[dict, dict]:
    service = ProcessService(settings)
    instance = service.create_instance(
        name="RGB POC",
        template_id="mining_steel_ball_classification_2d_reflectance_mvp",
        supported_input_type="rgb_image",
    )
    recipe = service.promote_recipe(instance["id"], run_id=None, recipe_type="production")
    return instance, recipe


def test_create_and_resolve_binding(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _, recipe = _recipe(settings)
    created = create_process_binding(
        CreateProcessBindingRequest(
            name="USB0 RGB manual",
            source_id="usb_camera_0",
            modality="rgb",
            purpose="manual_debug",
            pipeline_id="mining_steel_ball_classification_2d",
            active_recipe_version_id=recipe["id"],
            enabled=True,
        ),
        settings,
    )
    assert created["id"].startswith("binding_")

    resolved = ProcessBindingService(settings).resolve_process_binding("usb_camera_0", "rgb", "manual_debug")
    assert resolved is not None
    assert resolved["pipeline_id"] == "mining_steel_ball_classification_2d"

    listed = list_process_bindings(settings)
    assert len(listed) == 1


def test_reject_incompatible_modality(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _, recipe = _recipe(settings)
    with pytest.raises(Exception):
        create_process_binding(
            CreateProcessBindingRequest(
                name="Bad modality",
                source_id="usb_camera_0",
                modality="point_cloud",
                purpose="manual_debug",
                pipeline_id="mining_steel_ball_classification_2d",
                active_recipe_version_id=recipe["id"],
                enabled=True,
            ),
            settings,
        )


def test_reject_ambiguous_enabled_bindings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _, recipe = _recipe(settings)
    service = ProcessBindingService(settings)
    service.create_binding(
        name="b1",
        source_id="usb_camera_0",
        modality="rgb",
        purpose="manual_debug",
        pipeline_id="mining_steel_ball_classification_2d",
        active_recipe_version_id=recipe["id"],
        calibration_profile_id=None,
        enabled=True,
    )
    with pytest.raises(ValueError):
        service.create_binding(
            name="b2",
            source_id="usb_camera_0",
            modality="rgb",
            purpose="manual_debug",
            pipeline_id="mining_steel_ball_classification_2d",
            active_recipe_version_id=recipe["id"],
            calibration_profile_id=None,
            enabled=True,
        )


def test_run_stores_recipe_version_and_config_snapshot_hash(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _rgb_take(settings)
    _, recipe = _recipe(settings)
    service = ProcessBindingService(settings)
    binding = service.create_binding(
        name="run binding",
        source_id="usb_camera_0",
        modality="rgb",
        purpose="manual_debug",
        pipeline_id="mining_steel_ball_classification_2d",
        active_recipe_version_id=recipe["id"],
        calibration_profile_id=None,
        enabled=True,
    )

    process_take_for_pipeline(
        "take_rgb_bindings",
        ExecuteTakeRequest(pipeline_id="mining_steel_ball_classification_2d", source_id="usb_camera_0", modality="rgb", purpose="manual_debug"),
        settings,
    )
    # Runs live in the catalog now, not in processes/index/runs.json, which is
    # only a mirror refreshed on rebuild. Read them through the public accessor.
    from vision_3d_acquisition.processing.status_index import load_process_entries

    latest = load_process_entries(settings.data_dir)[-1]
    assert latest["pipeline_id"] == binding["pipeline_id"]
    assert latest["recipe_version_id"] == recipe["id"]
    assert isinstance(latest["config_snapshot_hash"], str) and len(latest["config_snapshot_hash"]) == 64


def test_manual_processing_still_works_without_binding(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _rgb_take(settings, "take_rgb_no_binding")
    response = process_take_for_pipeline(
        "take_rgb_no_binding",
        ExecuteTakeRequest(pipeline_id="mining_steel_ball_classification_2d", source_id="usb_camera_0", modality="rgb", purpose="manual_debug"),
        settings,
    )
    assert response["ok"] is True
    assert "warning" in response


def test_activate_recipe_endpoint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _, recipe = _recipe(settings)
    _, recipe2 = _recipe(settings)
    service = ProcessBindingService(settings)
    binding = service.create_binding(
        name="b",
        source_id="usb_camera_0",
        modality="rgb",
        purpose="manual_debug",
        pipeline_id="mining_steel_ball_classification_2d",
        active_recipe_version_id=recipe["id"],
        calibration_profile_id=None,
        enabled=True,
    )
    from vision_3d_acquisition.api.main import activate_process_binding_recipe

    updated = activate_process_binding_recipe(binding["id"], ActivateProcessBindingRecipeRequest(recipe_version_id=recipe2["id"]), settings)
    assert updated["active_recipe_version_id"] == recipe2["id"]
