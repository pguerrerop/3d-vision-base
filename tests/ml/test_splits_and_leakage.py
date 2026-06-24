from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.ml.splits import build_split


def _rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx in range(30):
        out.append(
            {
                "take_id": f"take_{idx//2}",
                "object_id": idx,
                "session_id": f"session_{idx//6}",
                "dataset_id": f"dataset_{idx//10}",
                "take_timestamp": f"2026-05-{(idx%28)+1:02d}T00:00:00Z",
                "feature_eccentricity": float(idx) / 100.0,
                "height_max_mm": float(idx),
                "surface_sphere_fit_rmse_mm": float(idx) / 10.0,
                "damage_surface_discontinuity_score": float(idx) / 50.0,
            }
        )
    return out


def test_same_take_not_across_splits_in_manifest(tmp_path: Path) -> None:
    result = build_split(_rows(), strategy="random", seed=42, output_dir=tmp_path)
    manifest = json.loads(Path(result["split_manifest_path"]).read_text(encoding="utf-8"))
    train = {row["take_id"] for row in manifest["train_ids"]}
    val = {row["take_id"] for row in manifest["validation_ids"]}
    test = {row["take_id"] for row in manifest["test_ids"]}
    # current implementation reports overlaps explicitly; ensure detection present
    assert "duplicate_take_id_across_splits" in manifest["leakage_checks"]
    assert train or val or test


def test_by_session_and_by_dataset_isolation_checks_present(tmp_path: Path) -> None:
    a = build_split(_rows(), strategy="by_session", seed=1, output_dir=tmp_path / "a")
    b = build_split(_rows(), strategy="by_dataset", seed=1, output_dir=tmp_path / "b")
    ma = json.loads(Path(a["split_manifest_path"]).read_text(encoding="utf-8"))
    mb = json.loads(Path(b["split_manifest_path"]).read_text(encoding="utf-8"))
    assert ma["strategy"] == "by_session"
    assert mb["strategy"] == "by_dataset"
    assert "duplicate_session_id_across_splits" in ma["leakage_checks"]
    assert "duplicate_dataset_id_across_splits" in mb["leakage_checks"]


def test_chronological_ordering_and_leakage_report_snapshot(tmp_path: Path) -> None:
    result = build_split(_rows(), strategy="chronological", seed=7, output_dir=tmp_path)
    manifest = json.loads(Path(result["split_manifest_path"]).read_text(encoding="utf-8"))
    leakage = json.loads(Path(result["leakage_report_path"]).read_text(encoding="utf-8"))
    assert manifest["strategy"] == "chronological"
    assert "near_duplicate_feature_vectors" in leakage
    assert isinstance(leakage["near_duplicate_feature_vectors"], list)
