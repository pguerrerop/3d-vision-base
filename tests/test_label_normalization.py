from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.datasets import DatasetService, LabelNormalizationService


def test_label_normalization_maps_known_tags_and_preserves_unknown() -> None:
    service = LabelNormalizationService(taxonomy_name="mining_balls_v1")
    result = service.normalize_tags(["ok 3", "bola chip", "desconocido"]) 
    assert result.semantic_labels == ["BALL_GOOD_LARGE", "BALL_CHIPPED"]
    assert result.superclass_labels == ["BALL_GOOD", "BALL_DEFECT"]
    assert result.normalized_class == "BALL_GOOD_LARGE"
    assert result.normalization_version == "v1"
    assert result.unknown_tags == ["desconocido"]


def test_label_normalization_is_deterministic() -> None:
    service = LabelNormalizationService(taxonomy_name="mining_balls_v1")
    a = service.normalize_tags(["mitad", "ok"]).normalized_class
    b = service.normalize_tags(["mitad", "ok"]).normalized_class
    assert a == b == "BALL_BROKEN"


def test_label_normalization_repairs_known_split_tag_pair() -> None:
    service = LabelNormalizationService(taxonomy_name="mining_balls_v1")
    result = service.normalize_tags(["ok 2 (2", "5\")"])
    assert result.semantic_labels == ["BALL_GOOD_MEDIUM"]
    assert result.superclass_labels == ["BALL_GOOD"]
    assert result.repaired_split_tags
    assert "ok 2 (2 + 5\") -> ok 2 (2,5\")" in result.repaired_split_tags


def test_observed_dataset_tags_are_covered() -> None:
    service = LabelNormalizationService(taxonomy_name="mining_balls_v1")
    observed = [
        "chica 2", "encoder", "80%", "ahuevada", "chica", "cubo", "mitad 2",
        "golilla", "tarro curry", "tarro spray", "zoquete", "cámara",
    ]
    result = service.normalize_tags(observed)
    assert result.unknown_tags == []


def test_apply_to_take_metadata_preserves_raw_tags() -> None:
    service = LabelNormalizationService(taxonomy_name="mining_balls_v1")
    payload = {"take_id": "t1", "tags": ["ok 2 (2", "5\")"]}
    out = service.apply_to_take_metadata(payload)
    assert out["tags"] == ["ok 2 (2", "5\")"]
    assert out["semantic_labels"] == ["BALL_GOOD_MEDIUM"]


def test_dataset_label_summary_aggregates_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ds = DatasetService(data_dir)
    ds.create_dataset(dataset_id="d1", name="Dataset 1")
    ds.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    ds.upsert_take_metadata(
        take_id="take_a",
        dataset_id="d1",
        session_id="s1",
        updates={
            "tags": ["ok", "perno"],
            "semantic_labels": ["BALL_GOOD_STANDARD", "SCRAP_BOLT"],
            "superclass_labels": ["BALL_GOOD", "SCRAP"],
            "normalized_class": "BALL_GOOD_STANDARD",
            "normalization_version": "v1",
        },
        source_metadata={},
    )
    ds.upsert_take_metadata(
        take_id="take_b",
        dataset_id="d1",
        session_id="s1",
        updates={
            "tags": ["misterio"],
            "normalization_warnings": ["UNMAPPED_TAG:misterio"],
            "normalization_version": "v1",
        },
        source_metadata={},
    )
    summary = ds.label_summary("d1")
    assert summary["raw_tag_counts"]["ok"] == 1
    assert summary["semantic_label_counts"]["BALL_GOOD_STANDARD"] == 1
    assert summary["superclass_counts"]["SCRAP"] == 1
    assert summary["unmapped_tags"]["misterio"] == 1
    assert summary["normalization_version"] == "v1"
