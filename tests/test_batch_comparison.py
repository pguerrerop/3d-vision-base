from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.batch_comparison import (
    compare_pipeline_sources_batch,
    get_pipeline_batch_comparison,
    list_pipeline_batch_comparisons,
)
from vision_3d_acquisition.processing.status_index import append_process_run_index

PIPELINE_ID = "mining_steel_ball_classification_25d"


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


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_mask(path: Path, rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rows, dtype=np.uint8) * 255, mode="L").save(path)


def _register_run(
    settings: ApiSettings,
    *,
    take_id: str,
    run_id: str,
    mask_rows: list[list[int]],
    object_count: int,
    label: str,
    superclass: str,
    confidence: float,
    created_at: str,
) -> None:
    # run_id is a global primary key in the process-run catalog (see
    # vision_3d_acquisition/storage/run_store.py: "Insert or replace one run
    # row"), so every run registered anywhere in a test must use a distinct
    # run_id -- reusing a literal id like "latest" across takes silently
    # clobbers earlier entries instead of raising.
    run_dir = settings.data_dir / "processes" / "runs" / run_id
    write_mask(run_dir / "final_object_mask.png", mask_rows)
    payload = {
        "take_id": take_id,
        "processed_at": created_at,
        "status": "completed",
        "processing_pipeline": {"id": PIPELINE_ID, "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
        ],
        "objects": [{"object_id": i} for i in range(1, object_count + 1)],
        "classification": {"label": label, "superclass": superclass, "confidence": confidence},
    }
    write_result(run_dir / "result.json", payload)
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id=f"instance_{take_id}",
        run_id=run_id,
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at=created_at,
        pipeline_id=PIPELINE_ID,
    )


def test_batch_comparison_aggregates_across_takes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    # Approved reference run, shared as the fixed left side for every take.
    _register_run(
        settings,
        take_id="take_ref",
        run_id="reference_run",
        mask_rows=[[1, 0], [0, 1]],
        object_count=1,
        label="accept",
        superclass="good",
        confidence=0.8,
        created_at="2026-07-01T09:00:00Z",
    )
    # take_unchanged: identical mask, no classification/object-count change.
    _register_run(
        settings,
        take_id="take_unchanged",
        run_id="run_take_unchanged",
        mask_rows=[[1, 0], [0, 1]],
        object_count=1,
        label="accept",
        superclass="good",
        confidence=0.8,
        created_at="2026-07-01T10:00:00Z",
    )
    # take_flip: mask changed, classification flips, object count changes.
    _register_run(
        settings,
        take_id="take_flip",
        run_id="run_take_flip",
        mask_rows=[[1, 1], [1, 0]],
        object_count=2,
        label="reject",
        superclass="bad",
        confidence=0.6,
        created_at="2026-07-01T10:05:00Z",
    )
    # take_mask_only: mask changed, classification/object count unchanged.
    _register_run(
        settings,
        take_id="take_mask_only",
        run_id="run_take_mask_only",
        mask_rows=[[1, 0], [1, 1]],
        object_count=1,
        label="accept",
        superclass="good",
        confidence=0.8,
        created_at="2026-07-01T10:10:00Z",
    )

    result = compare_pipeline_sources_batch(
        settings,
        pipeline_id=PIPELINE_ID,
        take_ids=["take_unchanged", "take_flip", "take_mask_only", "take_missing"],
        left={"type": "run", "take_id": "take_ref", "run_id": "reference_run"},
        right={"type": "run", "run_id": "latest"},
    )

    assert result["requested_take_count"] == 4
    assert result["compared_take_count"] == 3
    assert result["error_take_count"] == 1
    assert result["errors"] == [{"take_id": "take_missing", "error": "Unknown run for take take_missing: latest"}]
    assert {entry["take_id"] for entry in result["per_take"]} == {"take_unchanged", "take_flip", "take_mask_only"}

    aggregate = result["aggregate"]
    assert aggregate["compared_take_count"] == 3
    assert aggregate["classification_changed_count"] == 1
    assert aggregate["classification_changed_fraction"] == 1 / 3
    assert aggregate["object_count_changed_count"] == 1
    assert aggregate["takes_with_mask_changes_count"] == 2
    assert aggregate["takes_with_mask_changes_fraction"] == 2 / 3
    assert aggregate["min_iou"] < 1.0
    assert aggregate["max_iou"] == 1.0
    assert aggregate["average_iou"] is not None

    top_unit = aggregate["most_changed_units"][0]
    assert top_unit["unit_id"] == "remove_belt_segment_objects"
    assert top_unit["changed_take_count"] == 2
    assert top_unit["changed_take_fraction"] == 2 / 3

    # Persisted and listable like single comparisons.
    stored = get_pipeline_batch_comparison(settings, PIPELINE_ID, result["batch_comparison_id"])
    assert stored["batch_comparison_id"] == result["batch_comparison_id"]
    listed = list_pipeline_batch_comparisons(settings, PIPELINE_ID)
    assert listed[0]["batch_comparison_id"] == result["batch_comparison_id"]


def test_batch_comparison_left_take_id_is_not_overwritten_by_fill_in(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    _register_run(
        settings,
        take_id="take_a",
        run_id="take_a_reference",
        mask_rows=[[1, 0], [0, 1]],
        object_count=1,
        label="accept",
        superclass="good",
        confidence=0.8,
        created_at="2026-07-01T09:00:00Z",
    )
    _register_run(
        settings,
        take_id="take_a",
        run_id="take_a_latest",
        mask_rows=[[1, 0], [0, 1]],
        object_count=1,
        label="accept",
        superclass="good",
        confidence=0.8,
        created_at="2026-07-01T10:00:00Z",
    )
    _register_run(
        settings,
        take_id="take_b",
        run_id="take_b_latest",
        mask_rows=[[1, 1], [0, 0]],
        object_count=1,
        label="accept",
        superclass="good",
        confidence=0.8,
        created_at="2026-07-01T10:00:00Z",
    )

    # left pins an explicit take_id, so both takes get compared against the SAME
    # reference run rather than their own -- exercises that an explicit take_id
    # in the template is not overwritten by the per-take fill-in.
    result = compare_pipeline_sources_batch(
        settings,
        pipeline_id=PIPELINE_ID,
        take_ids=["take_a", "take_b"],
        left={"type": "run", "take_id": "take_a", "run_id": "take_a_reference"},
        right={"type": "run", "run_id": "latest"},
    )

    assert result["compared_take_count"] == 2
    for entry in result["per_take"]:
        comparison = get_pipeline_batch_comparison(settings, PIPELINE_ID, result["batch_comparison_id"])
        assert comparison["left"] == {"type": "run", "take_id": "take_a", "run_id": "take_a_reference"}
    assert {entry["take_id"] for entry in result["per_take"]} == {"take_a", "take_b"}
