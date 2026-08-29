from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vision_3d_acquisition.api.main import (
    ParentRunResultRefRequest,
    PartialRerunExecuteOptionsRequest,
    PartialRerunExecuteRequest,
    PartialRerunPlanRequest,
    pipeline_partial_rerun_execute,
    pipeline_partial_rerun_plan,
)
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.pipelines.run_lineage import (
    generate_run_comparison_to_parent,
    list_pipeline_runs,
    resolve_run_children,
    resolve_run_comparison_id,
    resolve_run_parent,
)
from vision_3d_acquisition.processing.status_index import append_process_run_index, update_process_run_index_entry
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from tests.test_pipeline_comparison import make_settings

PIPELINE_ID = "mining_steel_ball_classification_25d"


def _index_full_run(data_dir: Path, *, take_id: str, run_id: str, created_at: str) -> None:
    run_dir = data_dir / "processes" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    append_process_run_index(
        data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d",
        run_id=run_id,
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at=created_at,
        pipeline_id=PIPELINE_ID,
    )


def _index_partial_run(data_dir: Path, *, take_id: str, run_id: str, parent_run_id: str, created_at: str, boundary: str = "classification") -> None:
    run_dir = data_dir / "processes" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    append_process_run_index(
        data_dir,
        take_id=take_id,
        pipeline_instance_id="partial_rerun_25d",
        run_id=run_id,
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at=created_at,
        pipeline_id=PIPELINE_ID,
        execution_mode="rerun_from_public_stage_boundary",
        parent_run_id=parent_run_id,
        parent_take_id=take_id,
        boundary_stage_id=boundary,
        boundary_unit_id=boundary,
        run_summary={"reused_units": 5, "rerun_units": 2},
    )


def test_list_pipeline_runs_returns_full_and_partial_entries_sorted_desc(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _index_full_run(settings.data_dir, take_id="take_1", run_id="run_a", created_at="2026-01-01T00:00:00Z")
    _index_partial_run(settings.data_dir, take_id="take_1", run_id="run_b", parent_run_id="run_a", created_at="2026-01-02T00:00:00Z")

    runs = list_pipeline_runs(settings, PIPELINE_ID, take_id="take_1")

    assert [entry["run_id"] for entry in runs] == ["run_b", "run_a"]
    assert runs[0]["run_type"] == "partial_rerun"
    assert runs[1]["run_type"] == "full_run"


def test_list_pipeline_runs_filters_by_take_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _index_full_run(settings.data_dir, take_id="take_1", run_id="run_a", created_at="2026-01-01T00:00:00Z")
    _index_full_run(settings.data_dir, take_id="take_2", run_id="run_c", created_at="2026-01-03T00:00:00Z")

    runs_for_take_1 = list_pipeline_runs(settings, PIPELINE_ID, take_id="take_1")
    all_runs = list_pipeline_runs(settings, PIPELINE_ID)

    assert [entry["run_id"] for entry in runs_for_take_1] == ["run_a"]
    assert {entry["run_id"] for entry in all_runs} == {"run_a", "run_c"}


def test_resolve_run_parent_returns_parent_entry_for_child(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _index_full_run(settings.data_dir, take_id="take_1", run_id="run_a", created_at="2026-01-01T00:00:00Z")
    _index_partial_run(settings.data_dir, take_id="take_1", run_id="run_b", parent_run_id="run_a", created_at="2026-01-02T00:00:00Z")

    parent = resolve_run_parent(settings, PIPELINE_ID, "run_b")
    no_parent = resolve_run_parent(settings, PIPELINE_ID, "run_a")

    assert parent is not None
    assert parent["run_id"] == "run_a"
    assert no_parent is None


def test_resolve_run_children_returns_child_for_parent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _index_full_run(settings.data_dir, take_id="take_1", run_id="run_a", created_at="2026-01-01T00:00:00Z")
    _index_partial_run(settings.data_dir, take_id="take_1", run_id="run_b", parent_run_id="run_a", created_at="2026-01-02T00:00:00Z")
    _index_partial_run(settings.data_dir, take_id="take_1", run_id="run_c", parent_run_id="run_a", created_at="2026-01-03T00:00:00Z")

    children = resolve_run_children(settings, PIPELINE_ID, "run_a")

    assert [entry["run_id"] for entry in children] == ["run_c", "run_b"]
    assert all(entry["run_type"] == "partial_rerun" for entry in children)


def test_resolve_run_comparison_id_returns_id_when_present(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _index_partial_run(settings.data_dir, take_id="take_1", run_id="run_b", parent_run_id="run_a", created_at="2026-01-02T00:00:00Z")

    assert resolve_run_comparison_id(settings, PIPELINE_ID, "run_b") is None

    update_process_run_index_entry(settings.data_dir, "run_b", comparison_id="cmp_123")

    assert resolve_run_comparison_id(settings, PIPELINE_ID, "run_b") == "cmp_123"


def test_generate_run_comparison_to_parent_backfills_missing_comparison(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id="generate_comparison_now")
    full_result = run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)

    parent_run_id = "parent_for_generate_comparison"
    parent_run_dir = settings.data_dir / "processes" / "runs" / parent_run_id
    shutil.copytree(full_result.output_dir, parent_run_dir, dirs_exist_ok=True)
    parent_result_path = parent_run_dir / "result.json"
    parent_payload = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_payload["run_id"] = parent_run_id
    parent_result_path.write_text(json.dumps(parent_payload, indent=2), encoding="utf-8")
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d",
        run_id=parent_run_id,
        pipeline_family="25d",
        status="completed",
        run_dir=parent_run_dir,
        created_at=str(parent_payload.get("processed_at") or "2026-07-03T12:00:00Z"),
        pipeline_id=PIPELINE_ID,
    )

    snapshot = {
        "parameters_by_unit": {
            "remove_belt_segment_objects.foreground_thresholding": {"min_height_mm": 4.0},
        },
    }
    plan = pipeline_partial_rerun_plan(
        PIPELINE_ID,
        PartialRerunPlanRequest(
            selected_unit_id="remove_belt_segment_objects.foreground_thresholding",
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
            plan_id=str(plan["plan_id"]),
            parent_run_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
            effective_recipe_snapshot=snapshot,
            overrides={"parameters_by_unit": snapshot["parameters_by_unit"]},
            options=PartialRerunExecuteOptionsRequest(compare_to_parent=False),
        ),
        settings,
    )
    assert execution["ok"] is True
    child_run_id = execution["child_run_id"]
    assert execution["comparison_id"] is None

    # Simulate a "legacy" child run that never got a comparison_id at execute time.
    assert resolve_run_comparison_id(settings, PIPELINE_ID, child_run_id) is None

    comparison = generate_run_comparison_to_parent(settings, PIPELINE_ID, child_run_id)

    assert comparison["comparison_id"]
    assert resolve_run_comparison_id(settings, PIPELINE_ID, child_run_id) == comparison["comparison_id"]

    # Calling again should return the same comparison rather than duplicating it.
    comparison_again = generate_run_comparison_to_parent(settings, PIPELINE_ID, child_run_id)
    assert comparison_again["comparison_id"] == comparison["comparison_id"]


def test_generate_run_comparison_to_parent_raises_for_run_without_parent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _index_full_run(settings.data_dir, take_id="take_1", run_id="run_a", created_at="2026-01-01T00:00:00Z")

    with pytest.raises(ValueError):
        generate_run_comparison_to_parent(settings, PIPELINE_ID, "run_a")


def test_generate_run_comparison_to_parent_raises_for_missing_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(KeyError):
        generate_run_comparison_to_parent(settings, PIPELINE_ID, "does_not_exist")
