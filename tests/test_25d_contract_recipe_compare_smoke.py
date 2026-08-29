from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.debug.contract_recipe_compare_smoke import run_25d_contract_recipe_compare_smoke
from vision_3d_acquisition.pipelines.comparison import get_pipeline_comparison, list_pipeline_comparisons
from vision_3d_acquisition.pipelines.processing_units import validate_processing_unit_contracts
from vision_3d_acquisition.pipelines.recipes import RecipeService
from vision_3d_acquisition.pipelines.registry import list_processing_unit_definitions
from tests.test_pipeline_recipes import make_settings


def test_25d_contract_recipe_compare_smoke(tmp_path: Path) -> None:
    summary = run_25d_contract_recipe_compare_smoke(tmp_path)
    assert summary["ok"] is True
    assert summary["contract_validation"]["ok"] is True
    assert summary["mask_pixel_diff"]["changed_percent"] > 0
    assert summary["comparison_index_count"] >= 2
    assert summary["comparison_index_count_for_take"] >= 1

    settings = make_settings(tmp_path)
    listed = list_pipeline_comparisons(settings, "mining_steel_ball_classification_25d", take_id="take_smoke_compare")
    assert any(item["comparison_id"] == summary["run_compare_id"] for item in listed)
    all_listed = list_pipeline_comparisons(settings, "mining_steel_ball_classification_25d")
    assert any(item["comparison_id"] == summary["recipe_compare_id"] for item in all_listed)
    loaded = get_pipeline_comparison(settings, "mining_steel_ball_classification_25d", summary["run_compare_id"])
    assert loaded["summary"]["what_changed_most"] is not None


def test_recipe_fingerprint_mismatch_warns(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = RecipeService(settings)
    recipe = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="Fingerprint test",
        stage_params={"remove_belt_segment_objects": {"min_height_mm": 2.0}},
    )
    recipe["processing_unit_contract_fingerprint"] = "deadbeef"
    path = settings.data_dir / "recipes" / recipe["pipeline_id"] / recipe["recipe_id"] / "recipe.json"
    path.write_text(__import__("json").dumps(recipe, indent=2), encoding="utf-8")
    loaded = service.get_recipe(recipe["pipeline_id"], recipe["recipe_id"])
    assert loaded["compatibility"] == "compatible_with_warnings"
    assert any("fingerprint mismatch" in warning.lower() for warning in loaded["validation_warnings"])


def test_contract_validation_catches_duplicate_views_and_bad_diffable() -> None:
    bad_units = [
        {
            "id": "stage_a",
            "kind": "stage",
            "parent_id": None,
            "stage_id": "stage_a",
            "order": 0,
            "parameters": [],
            "artifacts": [
                {
                    "artifact_id": "mask_a",
                    "renderer": "image",
                    "diffable": True,
                }
            ],
            "views": [
                {"id": "main", "artifact_ids": ["mask_a"], "renderer_type": "image"},
                {"id": "main", "artifact_ids": ["mask_a"], "renderer_type": "image"},
            ],
            "default_view": "main",
        }
    ]
    validation = validate_processing_unit_contracts(bad_units)
    assert validation["ok"] is False
    assert any("duplicate view id" in error for error in validation["errors"])
    assert any("diffable but missing a valid diff_type" in error for error in validation["errors"])


def test_current_25d_contract_still_validates() -> None:
    units = list_processing_unit_definitions("mining_steel_ball_classification_25d")
    validation = validate_processing_unit_contracts(units)
    assert validation["ok"] is True
