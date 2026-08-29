from __future__ import annotations

import json
import shutil
from pathlib import Path

from vision_3d_acquisition.api.filesystem import get_take_detail, safe_take_file
from vision_3d_acquisition.api.main import (
    ParentRunResultRefRequest,
    PartialRerunExecuteOptionsRequest,
    PartialRerunExecuteRequest,
    PartialRerunPlanRequest,
    pipeline_partial_rerun_execute,
    pipeline_partial_rerun_plan,
)
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.processing.status_index import append_process_run_index
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from tests.test_pipeline_comparison import make_settings


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
        pipeline_instance_id="instance_25d",
        run_id=run_id,
        pipeline_family="25d",
        status=str(payload.get("status") or "completed"),
        run_dir=run_dir,
        created_at=str(payload.get("processed_at") or "2026-07-03T12:00:00Z"),
        pipeline_id="mining_steel_ball_classification_25d",
    )
    return run_dir


def _execute_partial_rerun(settings, take_id: str, parent_run_id: str) -> dict:
    snapshot = {
        "parameters_by_unit": {
            "remove_belt_segment_objects.foreground_thresholding": {
                "min_height_mm": 4.0,
            },
        },
    }
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="remove_belt_segment_objects.foreground_thresholding",
            changed_parameters=snapshot["parameters_by_unit"],
            current_recipe_snapshot=snapshot,
            parent_run_result_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
        ),
        True,
        settings,
    )
    return pipeline_partial_rerun_execute(
        "mining_steel_ball_classification_25d",
        PartialRerunExecuteRequest(
            plan_id=str(plan["plan_id"]),
            parent_run_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
            effective_recipe_snapshot=snapshot,
            overrides={"parameters_by_unit": snapshot["parameters_by_unit"]},
            options=PartialRerunExecuteOptionsRequest(compare_to_parent=True),
        ),
        settings,
    )


def test_get_take_detail_with_run_id_loads_specific_run_not_latest(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id="run_scoped_detail")
    full_result = run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)
    _index_parent_run(settings.data_dir, take_id, run_id="parent_run_x", result_dir=full_result.output_dir)

    execution = _execute_partial_rerun(settings, take_id, "parent_run_x")
    assert execution["ok"] is True
    child_run_id = execution["child_run_id"]

    parent_detail = get_take_detail(settings, take_id, run_id="parent_run_x")
    child_detail = get_take_detail(settings, take_id, run_id=child_run_id)

    assert parent_detail is not None
    assert child_detail is not None
    assert parent_detail.result is not None
    assert child_detail.result is not None
    assert parent_detail.result.get("run_id") == "parent_run_x"
    assert child_detail.result.get("run_id") == child_run_id
    assert child_detail.result.get("parent_run_id") == "parent_run_x"
    assert child_detail.result.get("execution_mode") == "rerun_from_public_stage_boundary"


def test_get_take_detail_resolves_reused_artifact_paths_for_child_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id="run_scoped_artifacts")
    full_result = run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)
    _index_parent_run(settings.data_dir, take_id, run_id="parent_run_y", result_dir=full_result.output_dir)

    execution = _execute_partial_rerun(settings, take_id, "parent_run_y")
    assert execution["ok"] is True

    child_result_path = Path(str(execution["result_path"]))
    reused_dir = child_result_path.parent / "reused" / "detect_belt_plane"
    assert reused_dir.exists()
    reused_files = list(reused_dir.glob("*"))
    assert reused_files, "expected at least one copied reused artifact"

    relative = reused_files[0].relative_to(child_result_path.parent)
    resolved = safe_take_file(settings, take_id, str(relative))
    assert resolved is not None
    assert resolved.is_file()


def test_get_take_detail_without_run_id_preserves_legacy_behavior(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id="run_scoped_legacy")
    run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)

    detail_without_run_id = get_take_detail(settings, take_id)
    detail_with_latest = get_take_detail(settings, take_id, run_id="latest")

    assert detail_without_run_id is not None
    assert detail_with_latest is not None
    assert detail_without_run_id.result is not None
    assert detail_with_latest.result is not None
    assert detail_without_run_id.result.get("status") == detail_with_latest.result.get("status")
    assert detail_without_run_id.result.get("processed_at") == detail_with_latest.result.get("processed_at")
