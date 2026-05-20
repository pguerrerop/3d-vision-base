from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.processing.status_index import append_process_run_index, summarize_take_processing


def test_status_index_supports_family_specific_summaries(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    processed_take = data_dir / "processed" / "take_1"
    processed_take.mkdir(parents=True, exist_ok=True)
    (processed_take / "DONE").touch()

    append_process_run_index(
        data_dir,
        take_id="take_1",
        pipeline_instance_id="pipeline_rgb",
        run_id="run_1",
        pipeline_family="2d",
        status="success",
        run_dir=data_dir / "processes" / "runs" / "pipeline_rgb" / "run_1",
        created_at="2026-05-18T12:00:00Z",
    )

    summary = summarize_take_processing(data_dir, "take_1", has_3d_done=True, processed_dir=processed_take)
    by_family = {item["family"]: item for item in summary}

    assert by_family["3d"]["hasCompletedOutput"] is True
    assert by_family["2d"]["hasCompletedOutput"] is True


def test_status_index_reports_unprocessed_for_missing_family(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    processed_take = data_dir / "processed" / "take_2"
    processed_take.mkdir(parents=True, exist_ok=True)

    summary = summarize_take_processing(data_dir, "take_2", has_3d_done=False, processed_dir=processed_take)
    by_family = {item["family"]: item for item in summary}

    assert by_family["3d"]["status"] == "never_processed"
    assert by_family["2d"]["status"] == "never_processed"
