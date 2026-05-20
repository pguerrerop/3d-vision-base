from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.contracts.artifacts import normalize_processing_artifacts


def test_backfill_artifacts_from_legacy_result() -> None:
    result = {
        "take_id": "take_1",
        "processed_at": "2026-05-16T00:00:00Z",
        "files": {
            "debug_plane_segmentation": "debug_plane_segmentation.png",
            "debug_foreground": "debug_foreground_extraction.png",
            "debug_clusters": "debug_clusters.png",
            "projection_xy_topdown": "projection_xy_topdown.png",
        },
        "objects": [{"object_id": 1, "class_name": "ball"}],
        "rejected_objects": [{"object_id": 2, "class_name": "non_ball"}],
    }
    artifacts = normalize_processing_artifacts(result, output_dir=Path("/tmp"))
    ids = {item["artifact_id"] for item in artifacts}
    assert "plane_segmentation" in ids
    assert "classification_table" in ids
    assert "measurement_table" in ids
    assert "classification_object_1" in ids
    assert "measurement_object_2" in ids
    assert "result_payload" in ids
    assert "xy_topdown" in ids


def test_explicit_artifacts_win_without_duplicates() -> None:
    result = {
        "take_id": "take_2",
        "processed_at": "2026-05-16T00:00:00Z",
        "artifacts": [
            {
                "artifact_id": "classification_table",
                "stage_id": "classification",
                "kind": "table",
                "title": "Classification results",
                "preview_available": False,
                "metadata": {},
            }
        ],
        "objects": [{"object_id": 1, "class_name": "ball"}],
    }
    artifacts = normalize_processing_artifacts(result, output_dir=Path("/tmp"))
    ids = [item["artifact_id"] for item in artifacts]
    assert ids.count("classification_table") == 1


def test_overlay_artifacts_include_geometry_and_lineage() -> None:
    result = {
        "take_id": "take_3",
        "processed_at": "2026-05-16T00:00:00Z",
        "files": {"debug_clusters": "debug_clusters.png"},
        "objects": [
            {
                "object_id": 7,
                "class_name": "ball",
                "confidence": 0.72,
                "center_mm": [120.0, 88.0, 12.0],
                "dimensions_mm": [33.0, 31.0, 30.0],
            }
        ],
    }
    artifacts = normalize_processing_artifacts(result, output_dir=Path("/tmp"))
    bbox = next(item for item in artifacts if item["artifact_id"] == "segmentation_overlay_bbox_7")
    ellipse = next(item for item in artifacts if item["artifact_id"] == "classification_overlay_ellipse_7")
    assert bbox["kind"] == "overlay"
    assert bbox["overlay_type"] == "bbox"
    assert bbox["target_artifact_id"] == "foreground_clusters"
    assert bbox["coordinate_space"] == "plot_pixel"
    assert "x" in bbox["geometry"]
    assert bbox["source_artifact_ids"] == ["measurement_object_7"]
    assert ellipse["overlay_type"] == "ellipse"
    assert ellipse["object_id"] == 7


def test_world_coordinate_overlays_are_marked_approximate() -> None:
    result = {
        "take_id": "take_world",
        "processed_at": "2026-05-16T00:00:00Z",
        "artifacts": [
            {
                "artifact_id": "world_overlay",
                "stage_id": "classification",
                "kind": "overlay",
                "title": "World-space overlay",
                "overlay_type": "ellipse",
                "coordinate_space": "world_mm",
                "target_artifact_id": "debug_clusters",
                "geometry": {"cx": 11.0, "cy": 9.0, "rx": 2.0, "ry": 2.0},
                "style": {"stroke": "#00ff88"},
            }
        ],
    }
    artifacts = normalize_processing_artifacts(result, output_dir=Path("/tmp"))
    overlay = next(item for item in artifacts if item["artifact_id"] == "world_overlay")
    assert overlay["coordinate_space"] == "world_mm"
    assert overlay["approximate"] is True
    assert "Overlay is approximate: world coordinates projected onto static debug image." in overlay["overlay_warnings"]


def test_projection_overlay_target_uses_projection_pixel_for_projection_artifacts() -> None:
    result = {
        "take_id": "take_projection",
        "processed_at": "2026-05-16T00:00:00Z",
        "files": {"projection_xy_topdown": "projection_xy_topdown.png"},
        "objects": [{"object_id": 5, "class_name": "ball", "center_mm": [10.0, 20.0, 1.0]}],
    }
    artifacts = normalize_processing_artifacts(result, output_dir=Path("/tmp"))
    bbox = next(item for item in artifacts if item["artifact_id"] == "segmentation_overlay_bbox_5")
    assert bbox["target_artifact_id"] == "xy_topdown"
    assert bbox["coordinate_space"] == "projection_pixel"
