from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.filesystem import get_take_detail
from vision_3d_acquisition.api.main import ObjectAnnotationUpsertRequest, upsert_take_object_annotation
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


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


def write_take(settings: ApiSettings, take_id: str) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-19T10:00:00Z", "session_id": "session_a", "files": {"rgb": "rgb.png"}}),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def write_result_with_blob(settings: ApiSettings, take_id: str, rows: list[dict]) -> None:
    out_dir = settings.processed_dir / take_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "take_id": take_id,
        "processed_at": "2026-05-19T12:00:00Z",
        "status": "ok",
        "summary": {"object_count": len(rows), "ball_count": 0, "non_ball_count": 0, "decision": "review", "confidence": None},
        "objects": [],
        "files": {"overlay": None, "point_cloud": None},
        "timing_ms": {"load": 0, "segmentation": 0, "classification": 0, "total": 0},
        "error": None,
        "artifacts": [
            {
                "artifact_id": "blob_metrics",
                "stage_id": "blob_detection",
                "kind": "metric",
                "title": "blob metrics",
                "preview_available": False,
                "metadata": {"entries": rows},
            }
        ],
    }
    (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (out_dir / "DONE").touch()


def test_object_annotation_persistence_and_take_detail_loading(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_obj")
    write_result_with_blob(settings, "take_obj", [{"blob_id": 1, "bbox": [10, 10, 20, 20], "centroid": [20, 20]}])

    resp = upsert_take_object_annotation(
        "take_obj",
        ObjectAnnotationUpsertRequest(
            source_stage="blob_detection",
            source_artifact_id="blob_contours",
            candidate_id="1",
            labels=["ball", "worn_ball"],
            expected_class="ball",
            expected_diameter_mm=80,
            validation_status="accepted",
            notes="looks valid",
        ),
        settings,
    )

    detail = get_take_detail(settings, "take_obj")

    assert resp["ok"] is True
    assert detail is not None
    assert detail.object_annotations
    ann = detail.object_annotations[0]
    assert ann["candidate_id"] == "1"
    assert ann["matched_candidate_id"] == "1"
    assert ann["matched_by"] == "candidate_id"
    assert ann["expected_class"] == "ball"


def test_object_annotation_matching_fallback_by_bbox_or_centroid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_match")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="legacy", name="Legacy")
    service.create_session(dataset_id="legacy", session_id="session_a", name="Session A")
    service.upsert_object_annotation(
        take_id="take_match",
        dataset_id="legacy",
        session_id="session_a",
        annotation={
            "id": "object_001",
            "source_stage": "blob_detection",
            "source_artifact_id": "blob_contours",
            "candidate_id": "999",
            "bbox": [10, 10, 20, 20],
            "centroid": [20, 20],
            "labels": ["false_positive"],
            "validation_status": "needs_review",
        },
        source_metadata={"session_id": "session_a"},
    )
    write_result_with_blob(settings, "take_match", [{"blob_id": 7, "bbox": [12, 11, 20, 20], "centroid": [21, 20]}])

    detail = get_take_detail(settings, "take_match")

    assert detail is not None
    assert detail.object_annotations
    ann = detail.object_annotations[0]
    assert ann["candidate_id"] == "999"
    assert ann["matched_candidate_id"] == "7"
    assert ann["matched_by"] in {"bbox_iou", "nearest_centroid"}


def test_object_annotation_label_editing_for_selected_candidate(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings, "take_edit")
    write_result_with_blob(settings, "take_edit", [{"blob_id": 3, "bbox": [0, 0, 10, 10], "centroid": [5, 5]}])

    first = upsert_take_object_annotation(
        "take_edit",
        ObjectAnnotationUpsertRequest(
            id="object_003",
            source_stage="blob_detection",
            source_artifact_id="blob_contours",
            candidate_id="3",
            labels=["non_ball"],
            validation_status="unreviewed",
        ),
        settings,
    )
    upsert_take_object_annotation(
        "take_edit",
        ObjectAnnotationUpsertRequest(
            id="object_003",
            source_stage="blob_detection",
            source_artifact_id="blob_contours",
            candidate_id="3",
            labels=["ball", "worn_ball"],
            expected_class="ball",
            validation_status="accepted",
        ),
        settings,
    )

    detail = get_take_detail(settings, "take_edit")

    assert first["ok"] is True
    assert detail is not None
    ann = detail.object_annotations[0]
    assert set(ann["labels"]) == {"ball", "worn_ball"}
    assert ann["expected_class"] == "ball"
    assert ann["validation_status"] == "accepted"
