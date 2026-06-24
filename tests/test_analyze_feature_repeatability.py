from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_feature_repeatability import analyze_repeatability
from vision_3d_acquisition.datasets import DatasetService


def _write_take(data_dir: Path, take_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-25T10:00:00Z", "session_id": "session_live", "files": {"rgb": "rgb.png"}}),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def _write_processed(data_dir: Path, take_id: str, diameter_mm: float, circularity: float) -> None:
    processed = data_dir / "processed" / take_id
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-25T10:00:00Z",
                "run_id": f"run_{take_id}",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "pipeline_family": "25d"},
                "summary": {"object_count": 1, "valid_pixel_ratio": 0.96, "invalid_pixel_ratio": 0.04},
                "objects": [{"object_id": 1, "diameter_mm": diameter_mm, "circularity": circularity}],
            }
        ),
        encoding="utf-8",
    )


def test_repeatability_analysis_generates_summary_and_csvs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_ml_set(dataset_id="d1", ml_set_id="balls_v1", name="Balls", task_type="classification")

    for take_id, obj_id, diameter in [("t1", "obj_1", 84.2), ("t2", "obj_1", 84.3), ("t3", "obj_2", 83.9)]:
        _write_take(data_dir, take_id)
        service.upsert_take_metadata(
            take_id=take_id,
            dataset_id="d1",
            session_id="s1",
            updates={"physical_object_id": obj_id},
            source_metadata={"session_id": "session_live"},
        )
        service.add_take_to_ml_set(dataset_id="d1", ml_set_id="balls_v1", take_id=take_id, physical_object_id=obj_id, split="train")
        _write_processed(data_dir, take_id, diameter_mm=diameter, circularity=0.97)

    output_dir = data_dir / "runtime" / "analysis" / "repeatability_test"
    summary = analyze_repeatability(
        data_dir=data_dir,
        output_dir=output_dir,
        dataset_id="d1",
        ml_set_id="balls_v1",
        pipeline_id="mining_steel_ball_classification_25d",
        include_diagnostics=True,
        include_invalidity_flags=True,
        include_provenance_summaries=True,
    )

    assert summary["take_count_analyzed"] == 3
    assert summary["physical_object_count"] == 2
    assert (output_dir / "repeatability_summary.json").is_file()
    assert (output_dir / "repeatability_per_object_feature.csv").is_file()
    assert (output_dir / "repeatability_feature_stability.csv").is_file()
    assert (output_dir / "repeatability_correlations.csv").is_file()
    assert "quality_associations" in summary
    assert "provenance_validity" in summary
