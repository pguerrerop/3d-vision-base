from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.api.main import ExecuteTakeRequest, process_take_for_pipeline
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.processing_units import (
    recipe_parameters_by_unit,
    validate_processing_unit_contracts,
)
from vision_3d_acquisition.pipelines.recipes import RecipeService
from vision_3d_acquisition.pipelines.registry import list_processing_unit_definitions


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


def test_recipe_service_create_update_clone_archive(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = RecipeService(settings)
    stage_params = {
        "detect_belt_plane": {"background_detection_strategy": "low_gradient_depth_plateaus"},
        "remove_belt_segment_objects": {"min_height_mm": 2.0},
    }

    created = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="Conservative",
        stage_params=stage_params,
        provenance={"source": "test"},
    )
    assert created["recipe_id"] == "recipe_conservative"
    assert created["version"] == 1
    assert created["compatibility"] in {"compatible", "compatible_with_warnings"}

    listed = service.list_recipes("mining_steel_ball_classification_25d")
    assert [item["recipe_id"] for item in listed] == ["recipe_conservative"]

    fetched = service.get_recipe("mining_steel_ball_classification_25d", "recipe_conservative")
    assert fetched["stage_params"]["detect_belt_plane"]["background_detection_strategy"] == "low_gradient_depth_plateaus"

    updated = service.update_recipe(
        "mining_steel_ball_classification_25d",
        "recipe_conservative",
        name="Conservative",
        stage_params={
            **stage_params,
            "remove_belt_segment_objects": {"min_height_mm": 3.0},
        },
        provenance={"source": "update"},
    )
    assert updated["version"] == 2
    assert updated["stage_params"]["remove_belt_segment_objects"]["min_height_mm"] == 3.0

    cloned = service.clone_recipe("mining_steel_ball_classification_25d", "recipe_conservative", new_name="Conservative clone")
    assert cloned["parent_recipe_id"] == "recipe_conservative"
    assert cloned["version"] == 1

    archived = service.archive_recipe("mining_steel_ball_classification_25d", "recipe_conservative")
    assert archived["archived"] is True
    assert len(service.list_recipes("mining_steel_ball_classification_25d")) == 1


def test_recipe_validation_warns_for_unknown_legacy_params(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = RecipeService(settings)

    validation = service.validate_recipe(
        "mining_steel_ball_classification_25d",
        parameters_by_unit={
            "detect_belt_plane.height_gate": {
                "reference_surface_height_gate_enabled": True,
                "legacy_unknown_knob": 123,
            }
        },
    )

    assert validation["compatibility"] == "compatible_with_warnings"
    assert any("legacy_unknown_knob" in warning for warning in validation["warnings"])


def test_processing_unit_contract_validation_passes_for_current_25d_registry() -> None:
    units = list_processing_unit_definitions("mining_steel_ball_classification_25d")
    validation = validate_processing_unit_contracts(units)
    assert validation["ok"] is True
    assert validation["errors"] == []


def test_processing_unit_contract_validation_catches_bad_contracts() -> None:
    bad_units = [
        {
            "id": "stage_a",
            "kind": "stage",
            "parent_id": None,
            "stage_id": "stage_a",
            "parameters": [{"id": "alpha", "type": "number"}],
            "artifacts": [{"artifact_id": "a1"}],
            "views": [{"id": "main", "artifact_ids": ["missing_artifact"]}],
            "default_view": "unknown",
        },
        {
            "id": "stage_a",
            "kind": "substage",
            "parent_id": "missing_parent",
            "stage_id": "stage_b",
            "parameters": [{"id": "alpha", "type": "mystery"}],
            "artifacts": [{"artifact_id": "a1"}],
            "views": [],
            "default_view": None,
        },
    ]
    validation = validate_processing_unit_contracts(bad_units)
    assert validation["ok"] is False
    assert any("Duplicate processing unit id" in error for error in validation["errors"])
    assert any("missing parent_id" in error for error in validation["errors"])
    assert any("unsupported type mystery" in error for error in validation["errors"])
    assert any("default_view" in error for error in validation["errors"])


def test_process_take_applies_selected_recipe_metadata_and_overrides(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    take_dir = settings.incoming_dir / "take_recipe"
    take_dir.mkdir(parents=True, exist_ok=True)
    service = RecipeService(settings)
    recipe = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="Run recipe",
        stage_params={
            "detect_belt_plane": {"background_detection_strategy": "low_gradient_depth_plateaus"},
            "remove_belt_segment_objects": {"min_height_mm": 2.0},
        },
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "vision_3d_acquisition.api.main.get_take_detail",
        lambda settings_arg, take_id: type("Detail", (), {"metadata": {}, "modalities": ["heightmap"], "take_metadata": {}})(),
    )

    def fake_dispatch_take_processing(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("vision_3d_acquisition.api.main.dispatch_take_processing", fake_dispatch_take_processing)

    process_take_for_pipeline(
        "take_recipe",
        ExecuteTakeRequest(
            pipeline_id="mining_steel_ball_classification_25d",
            recipe_id=recipe["recipe_id"],
            recipe_version=recipe["version"],
            stage_params={"remove_belt_segment_objects": {"min_height_mm": 5.0}},
        ),
        settings,
    )

    assert captured["pipeline_id"] == "mining_steel_ball_classification_25d"
    assert captured["recipe_metadata"]["recipe_id"] == recipe["recipe_id"]
    assert captured["stage_params"]["detect_belt_plane"]["background_detection_strategy"] == "low_gradient_depth_plateaus"
    assert captured["stage_params"]["remove_belt_segment_objects"]["min_height_mm"] == 5.0


def test_recipe_parameters_by_unit_extracts_grouped_values() -> None:
    units = list_processing_unit_definitions("mining_steel_ball_classification_25d")
    grouped = recipe_parameters_by_unit(
        units,
        {
            "detect_belt_plane": {"background_detection_strategy": "low_gradient_surface"},
            "remove_belt_segment_objects": {"min_height_mm": 4.0},
        },
    )
    assert grouped["detect_belt_plane"]["background_detection_strategy"] == "low_gradient_surface"
    assert grouped["remove_belt_segment_objects"]["min_height_mm"] == 4.0


def test_recipe_service_accepts_low_gradient_bg_and_stripes_strategy(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = RecipeService(settings)
    recipe = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="BG and stripes",
        stage_params={
            "detect_belt_plane": {"background_detection_strategy": "low_gradient_bg_and_stripes"},
        },
    )
    assert recipe["stage_params"]["detect_belt_plane"]["background_detection_strategy"] == "low_gradient_bg_and_stripes"


def test_process_take_passes_low_gradient_bg_and_stripes_strategy_to_dispatch(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    take_dir = settings.incoming_dir / "take_bg_stripes"
    take_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "vision_3d_acquisition.api.main.get_take_detail",
        lambda settings_arg, take_id: type("Detail", (), {"metadata": {}, "modalities": ["heightmap"], "take_metadata": {}})(),
    )

    def fake_dispatch_take_processing(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("vision_3d_acquisition.api.main.dispatch_take_processing", fake_dispatch_take_processing)

    process_take_for_pipeline(
        "take_bg_stripes",
        ExecuteTakeRequest(
            pipeline_id="mining_steel_ball_classification_25d",
            stage_params={"detect_belt_plane": {"background_detection_strategy": "low_gradient_bg_and_stripes"}},
        ),
        settings,
    )

    assert captured["stage_params"]["detect_belt_plane"]["background_detection_strategy"] == "low_gradient_bg_and_stripes"
