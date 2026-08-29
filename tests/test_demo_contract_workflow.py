from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.debug.demo_contract_workflow import run_25d_demo_contract_workflow


def test_demo_contract_workflow_produces_studio_links(tmp_path: Path) -> None:
    summary = run_25d_demo_contract_workflow(tmp_path)
    assert summary["ok"] is True
    assert summary["demo_take_id"] == "take_smoke_compare"
    assert "studio_open_url" in summary
    assert "graph_workspace=1" in summary["studio_open_url"]
    assert summary["run_compare_id"]
    assert summary["comparison_index_count"] >= 1
