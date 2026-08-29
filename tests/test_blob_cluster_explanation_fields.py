from __future__ import annotations

import csv
import json
from pathlib import Path

from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    assert matches, f"expected to find {name} under {root}"
    return matches[0]


def _run_blob_strategy(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="blob_explanation")
    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={"detect_belt_plane": {"background_detection_strategy": "low_gradient_blob_height_clusters"}},
    )
    return data_dir


def test_cluster_score_table_includes_selected_and_decision_columns(tmp_path: Path) -> None:
    data_dir = _run_blob_strategy(tmp_path)
    csv_path = _find_one(data_dir, "blob_cluster_score_table.csv")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for required in ("selected", "decision", "cluster_id", "score", "rejected_reason"):
        assert required in fieldnames, f"missing column {required}: {fieldnames}"

    if rows:
        # decision values are constrained, and at most one cluster is selected.
        decisions = {row["decision"] for row in rows}
        assert decisions <= {"selected", "rejected", "candidate"}
        selected_rows = [row for row in rows if str(row["selected"]).lower() == "true"]
        assert len(selected_rows) <= 1


def test_cluster_selection_debug_includes_membership_and_score_terms(tmp_path: Path) -> None:
    data_dir = _run_blob_strategy(tmp_path)
    debug_path = _find_one(data_dir, "blob_cluster_selection_debug.json")
    payload = json.loads(debug_path.read_text(encoding="utf-8"))

    assert "clusters" in payload, payload.keys()
    clusters = payload["clusters"]
    assert isinstance(clusters, list)
    for cluster in clusters:
        # additive membership + decision + score breakdown fields
        assert "component_ids" in cluster
        assert "fragment_ids" in cluster
        assert "decision" in cluster
        assert "selected" in cluster
        assert "score_terms" in cluster


def test_blob_height_clusters_rows_carry_decision_fields(tmp_path: Path) -> None:
    data_dir = _run_blob_strategy(tmp_path)
    clusters_path = _find_one(data_dir, "blob_height_clusters.json")
    payload = json.loads(clusters_path.read_text(encoding="utf-8"))

    # Active blob runs emit a list of cluster rows.
    if isinstance(payload, list) and payload:
        for row in payload:
            assert "selected" in row
            assert "decision" in row
            assert "component_ids" in row
            assert "fragment_ids" in row
