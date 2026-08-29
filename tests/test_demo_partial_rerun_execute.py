from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.debug.demo_partial_rerun_execute import run_25d_demo_partial_rerun_execute


def test_demo_partial_rerun_execute_verifies_lineage_manifest_and_comparison(tmp_path: Path) -> None:
    summary = run_25d_demo_partial_rerun_execute(tmp_path)

    assert summary["ok"] is True
    assert summary["child_run_id"]
    assert summary["comparison_id"]
    assert f"run_id={summary['child_run_id']}" in summary["graph_workspace_url"]

    verification = summary["verification"]
    assert verification["child_run_in_listing"] is True
    assert verification["lineage_resolves_parent"] is True
    assert verification["lineage_resolves_children"] is True
    assert verification["artifact_manifest_exists"] is True
    assert verification["comparison_available"] is True
    assert verification["graph_workspace_url_includes_child_run_id"] is True
