from __future__ import annotations

from vision_3d_acquisition.contracts.object_candidates import (
    extract_object_candidates_from_legacy_25d,
    normalize_object_candidates,
)
from vision_3d_acquisition.contracts.result import ObjectCandidate


def test_object_candidate_schema_serialization() -> None:
    candidate = ObjectCandidate(
        id="cand_1",
        local_object_index=1,
        source_modality="rgb",
        source_take_id="take_1",
        source_pipeline_run_id="run_1",
        acquisition_group_id=None,
        bbox_px=[1, 2, 3, 4],
        contour_px=[[1, 2], [3, 4]],
        mask_artifact_id="cleaned_mask",
        overlay_artifact_id="overlay",
        centroid_px=[2, 3],
        centroid_world=None,
        geometry={"area_px": 123},
        appearance={},
        measurements={},
        classification_hints={},
        confidence=0.8,
        diagnostics={},
    )
    dumped = candidate.model_dump(mode="json")
    assert dumped["id"] == "cand_1"
    assert dumped["source_modality"] == "rgb"


def test_extract_legacy_25d_candidates() -> None:
    payload = {
        "take_id": "take_25d",
        "input_modalities": ["heightmap"],
        "objects": [
            {
                "object_id": 1,
                "center_mm": [1.0, 2.0, 3.0],
                "bbox_px": [10, 20, 30, 40],
                "contour_px": [[10, 20], [11, 20], [11, 21]],
                "centroid_px": [12.5, 22.0],
                "height_above_belt_mm": {"max_height_mm": 9.5, "mean_height_mm": 4.2, "p95_height_mm": 8.8},
                "footprint_area_mm2": 40.0,
                "feature_volume_proxy_mm3": 100.0,
                "diameter_mm": 10.0,
                "class_name": "ball",
                "confidence": 0.9,
            }
        ],
        "artifacts": [{"artifact_id": "cleaned_mask"}, {"artifact_id": "classification_overlay"}],
    }
    rows = extract_object_candidates_from_legacy_25d(payload)
    assert len(rows) == 1
    assert rows[0]["source_modality"] == "derived_25d"
    assert rows[0]["bbox_px"] == [10, 20, 30, 40]


def test_empty_detections_emit_empty_list() -> None:
    payload = {"take_id": "take_empty", "input_modalities": ["rgb"], "artifacts": [], "objects": []}
    rows = normalize_object_candidates(payload)
    assert rows == []
