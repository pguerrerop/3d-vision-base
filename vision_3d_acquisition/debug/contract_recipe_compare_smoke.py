from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.comparison import compare_pipeline_sources, list_pipeline_comparisons
from vision_3d_acquisition.pipelines.processing_units import validate_processing_unit_contracts
from vision_3d_acquisition.pipelines.recipes import RecipeService
from vision_3d_acquisition.pipelines.registry import list_processing_unit_definitions
from vision_3d_acquisition.processing.status_index import append_process_run_index

PIPELINE_ID = "mining_steel_ball_classification_25d"


def _make_settings(data_dir: Path) -> ApiSettings:
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


def _write_mask(path: Path, rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rows, dtype=np.uint8) * 255, mode="L").save(path)


def run_25d_contract_recipe_compare_smoke(data_dir: Path) -> dict[str, Any]:
    settings = _make_settings(data_dir)
    units = list_processing_unit_definitions(PIPELINE_ID)
    contract_validation = validate_processing_unit_contracts(units)
    if not contract_validation["ok"]:
        raise RuntimeError(f"Contract validation failed: {contract_validation['errors']}")

    recipe_service = RecipeService(settings)
    try:
        base_recipe = recipe_service.create_recipe(
            PIPELINE_ID,
            name="Smoke baseline",
            recipe_id="recipe_smoke_baseline",
            stage_params={"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        )
    except ValueError:
        base_recipe = recipe_service.get_recipe(PIPELINE_ID, "recipe_smoke_baseline")
    try:
        clone_recipe = recipe_service.clone_recipe(
            PIPELINE_ID,
            base_recipe["recipe_id"],
            new_name="Smoke baseline clone",
            new_recipe_id="recipe_smoke_baseline_clone",
        )
    except ValueError:
        clone_recipe = recipe_service.get_recipe(PIPELINE_ID, "recipe_smoke_baseline_clone")
    recipe_compare = compare_pipeline_sources(
        settings,
        pipeline_id=PIPELINE_ID,
        left={"type": "recipe", "recipe_id": base_recipe["recipe_id"], "version": base_recipe["version"]},
        right={"type": "recipe", "recipe_id": clone_recipe["recipe_id"], "version": clone_recipe["version"]},
    )

    take_id = "take_smoke_compare"
    run_a_dir = settings.data_dir / "processes" / "runs" / "smoke_run_a"
    run_b_dir = settings.data_dir / "processes" / "runs" / "smoke_run_b"
    _write_mask(run_a_dir / "final_object_mask.png", [[1, 1], [0, 0]])
    _write_mask(run_b_dir / "final_object_mask.png", [[1, 1], [1, 0]])
    result_template = {
        "take_id": take_id,
        "status": "completed",
        "processing_pipeline": {"id": PIPELINE_ID, "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {
                "artifact_id": "final_object_mask",
                "stage_id": "remove_belt_segment_objects",
                "kind": "image",
                "title": "Mask",
                "path": "final_object_mask.png",
                "preview_available": True,
            }
        ],
        "objects": [{"object_id": 1}],
        "classification": {"label": "accept", "superclass": "good", "confidence": 0.8},
    }
    (run_a_dir / "result.json").write_text(json.dumps(result_template, indent=2), encoding="utf-8")
    result_b = {
        **result_template,
        "objects": [{"object_id": 1}, {"object_id": 2}],
        "classification": {"label": "reject", "superclass": "bad", "confidence": 0.6},
    }
    (run_b_dir / "result.json").write_text(json.dumps(result_b, indent=2), encoding="utf-8")
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d_smoke",
        run_id="smoke_run_a",
        pipeline_family="25d",
        status="completed",
        run_dir=run_a_dir,
        created_at="2026-07-01T10:00:00Z",
        pipeline_id=PIPELINE_ID,
    )
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d_smoke",
        run_id="smoke_run_b",
        pipeline_family="25d",
        status="completed",
        run_dir=run_b_dir,
        created_at="2026-07-01T10:05:00Z",
        pipeline_id=PIPELINE_ID,
    )
    run_compare = compare_pipeline_sources(
        settings,
        pipeline_id=PIPELINE_ID,
        left={"type": "run", "take_id": take_id, "run_id": "smoke_run_a"},
        right={"type": "run", "take_id": take_id, "run_id": "smoke_run_b"},
    )
    mask_row = next(
        row
        for row in run_compare["units"]["remove_belt_segment_objects"]["artifact_diff"]
        if row["artifact_id"] == "final_object_mask"
    )
    if not mask_row.get("pixel_diff"):
        raise RuntimeError("Expected mask pixel diff metrics for final_object_mask.")

    comparisons = list_pipeline_comparisons(settings, PIPELINE_ID, take_id=take_id, limit=5)
    all_comparisons = list_pipeline_comparisons(settings, PIPELINE_ID, limit=5)
    return {
        "ok": True,
        "pipeline_id": PIPELINE_ID,
        "contract_validation": contract_validation,
        "recipes": {
            "base_recipe_id": base_recipe["recipe_id"],
            "clone_recipe_id": clone_recipe["recipe_id"],
        },
        "recipe_compare_id": recipe_compare["comparison_id"],
        "run_compare_id": run_compare["comparison_id"],
        "mask_pixel_diff": mask_row["pixel_diff"],
        "what_changed_most": run_compare["summary"].get("what_changed_most"),
        "comparison_index_count": len(all_comparisons),
        "comparison_index_count_for_take": len(comparisons),
    }
