from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.api.main import ParentRunResultRefRequest, PartialRerunPlanRequest, pipeline_partial_rerun_plan
from vision_3d_acquisition.debug.demo_contract_workflow import DEMO_TAKE_ID, PIPELINE_ID, run_25d_demo_contract_workflow


def run_25d_demo_partial_rerun_plan(data_dir: Path) -> dict[str, Any]:
    base = run_25d_demo_contract_workflow(data_dir)
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
    scenarios = {
        "segmentation_threshold": {
            "selected_unit_id": "remove_belt_segment_objects.foreground_thresholding",
            "current_recipe_snapshot": {
                "parameters_by_unit": {
                    "remove_belt_segment_objects.foreground_thresholding": {"min_height_mm": 4.0},
                },
            },
        },
        "classification_unit": {
            "selected_unit_id": "classification.primary_heuristic_classifier",
            "current_recipe_snapshot": {
                "parameters_by_unit": {
                    "classification.primary_heuristic_classifier": {"confidence_threshold": 0.8},
                },
            },
        },
        "detect_reference_unit": {
            "selected_unit_id": "detect_belt_plane.height_gate",
            "current_recipe_snapshot": {
                "parameters_by_unit": {
                    "detect_belt_plane.height_gate": {"gradient_threshold_percentile": 75.0},
                },
            },
        },
    }
    plans: dict[str, Any] = {}
    for key, scenario in scenarios.items():
        plan = pipeline_partial_rerun_plan(
            PIPELINE_ID,
            PartialRerunPlanRequest(
                selected_unit_id=str(scenario["selected_unit_id"]),
                current_recipe_snapshot=dict(scenario["current_recipe_snapshot"]),
                parent_run_result_ref=ParentRunResultRefRequest(take_id=DEMO_TAKE_ID, run_id="smoke_run_b"),
            ),
            True,
            settings,
        )
        plans[key] = plan
    default_unit = plans["segmentation_threshold"]["selected_unit_id"]
    plan_id = plans["segmentation_threshold"].get("plan_id")
    query = (
        f"take_id={DEMO_TAKE_ID}"
        f"&pipeline_id={PIPELINE_ID}"
        f"&graph_workspace=1"
        f"&recipe_id={(base.get('recipes') or {}).get('base_recipe_id')}"
        f"&comparison_id={base.get('run_compare_id')}"
        f"&run_id=smoke_run_b"
        f"&unit_id={default_unit}"
    )
    return {
        **base,
        "partial_rerun_plans": plans,
        "graph_workspace_url": f"/studio?{query}",
        "planner_graph_url": f"/studio?{query}",
        "default_plan_id": plan_id,
        "next_steps": [
            "Open planner_graph_url",
            "Inspect the persisted partial rerun plan in Graph / Compare Workspace",
            "Compare stage-boundary elevation across segmentation, classification, and detect-reference units",
        ],
    }
