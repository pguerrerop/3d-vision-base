from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.main import (
    ParentRunResultRefRequest,
    PartialRerunExecuteOptionsRequest,
    PartialRerunExecuteRequest,
    PartialRerunPlanRequest,
    pipeline_partial_rerun_execute,
    pipeline_partial_rerun_plan,
)
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.pipelines.run_lineage import (
    generate_run_comparison_to_parent,
    list_pipeline_runs,
    resolve_run_children,
    resolve_run_parent,
)
from vision_3d_acquisition.processing.status_index import append_process_run_index
from vision_3d_acquisition.synthetic import create_synthetic_25d_take

PIPELINE_ID = "mining_steel_ball_classification_25d"


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


def _index_parent_run(data_dir: Path, take_id: str, *, run_id: str, result_dir: Path) -> Path:
    run_dir = data_dir / "processes" / "runs" / run_id
    shutil.copytree(result_dir, run_dir, dirs_exist_ok=True)
    result_path = run_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["run_id"] = run_id
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_process_run_index(
        data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d_demo_partial",
        run_id=run_id,
        pipeline_family="25d",
        status=str(payload.get("status") or "completed"),
        run_dir=run_dir,
        created_at=str(payload.get("processed_at") or "2026-07-03T16:00:00Z"),
        pipeline_id=PIPELINE_ID,
    )
    return run_dir


def _studio_url(**params: Any) -> str:
    query = "&".join(f"{key}={value}" for key, value in params.items() if value is not None)
    return f"/studio?{query}"


def run_25d_demo_partial_rerun_execute(data_dir: Path) -> dict[str, Any]:
    settings = _settings(data_dir)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id="demo_partial_rerun_execute")
    full_result = run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)
    parent_run_id = "smoke_partial_parent"
    _index_parent_run(settings.data_dir, take_id, run_id=parent_run_id, result_dir=full_result.output_dir)

    snapshot = {
        "parameters_by_unit": {
            "classification.primary_heuristic_classifier": {
                "confidence_threshold": 0.8,
            },
        },
    }
    plan = pipeline_partial_rerun_plan(
        PIPELINE_ID,
        PartialRerunPlanRequest(
            selected_unit_id="classification.primary_heuristic_classifier",
            changed_parameters=snapshot["parameters_by_unit"],
            current_recipe_snapshot=snapshot,
            parent_run_result_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
        ),
        True,
        settings,
    )
    execution = pipeline_partial_rerun_execute(
        PIPELINE_ID,
        PartialRerunExecuteRequest(
            plan_id=str(plan.get("plan_id") or ""),
            parent_run_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
            effective_recipe_snapshot=snapshot,
            overrides={"parameters_by_unit": snapshot["parameters_by_unit"]},
            options=PartialRerunExecuteOptionsRequest(compare_to_parent=True),
        ),
        settings,
    )
    child_run_id = execution.get("child_run_id")
    comparison_id = execution.get("comparison_id")
    graph_workspace_url = _studio_url(
        take_id=take_id,
        pipeline_id=PIPELINE_ID,
        graph_workspace=1,
        comparison_id=comparison_id,
        run_id=child_run_id,
        unit_id="classification.primary_heuristic_classifier",
    )
    parent_run_url = _studio_url(take_id=take_id, pipeline_id=PIPELINE_ID, run_id=parent_run_id)
    child_run_url = _studio_url(take_id=take_id, pipeline_id=PIPELINE_ID, run_id=child_run_id)
    comparison_url = (
        _studio_url(take_id=take_id, pipeline_id=PIPELINE_ID, run_id=child_run_id, comparison_id=comparison_id)
        if comparison_id
        else None
    )

    verification: dict[str, Any] = {
        "child_run_in_listing": False,
        "lineage_resolves_parent": False,
        "lineage_resolves_children": False,
        "artifact_manifest_exists": False,
        "comparison_available": False,
    }
    if child_run_id:
        runs = list_pipeline_runs(settings, PIPELINE_ID, take_id=take_id)
        verification["child_run_in_listing"] = any(run.get("run_id") == child_run_id for run in runs)

        parent_entry = resolve_run_parent(settings, PIPELINE_ID, child_run_id)
        verification["lineage_resolves_parent"] = bool(parent_entry and parent_entry.get("run_id") == parent_run_id)

        children_of_parent = resolve_run_children(settings, PIPELINE_ID, parent_run_id)
        verification["lineage_resolves_children"] = any(entry.get("run_id") == child_run_id for entry in children_of_parent)

        result_path = execution.get("result_path")
        if result_path:
            manifest_path = Path(str(result_path)).parent / "partial_rerun_artifact_manifest.json"
            verification["artifact_manifest_exists"] = manifest_path.is_file()

        if comparison_id:
            verification["comparison_available"] = True
        else:
            generated = generate_run_comparison_to_parent(settings, PIPELINE_ID, child_run_id)
            comparison_id = generated.get("comparison_id")
            verification["comparison_available"] = bool(comparison_id)

    verification["graph_workspace_url_includes_child_run_id"] = bool(child_run_id) and f"run_id={child_run_id}" in graph_workspace_url

    return {
        "ok": bool(execution.get("ok")),
        "pipeline_id": PIPELINE_ID,
        "demo_take_id": take_id,
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
        "comparison_id": comparison_id,
        "partial_rerun_plan_id": plan.get("plan_id"),
        "partial_rerun_plan": plan,
        "partial_rerun_execution": execution,
        "graph_workspace_url": graph_workspace_url,
        "parent_run_url": parent_run_url,
        "child_run_url": child_run_url,
        "comparison_url": comparison_url,
        "verification": verification,
    }
