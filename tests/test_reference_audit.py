from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import sensor_studio_cli as cli
from vision_3d_acquisition.debug.reference_audit import (
    ResolvedRun,
    build_reference_audit_bundle,
    compute_support_loss,
    load_binary_mask,
    mask_a_minus_b,
    mask_intersection,
    mask_union,
    overlay_diff,
    resolve_processed_run,
    save_binary_mask,
    _extract_derived_thresholds,
)
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow


def _write_mask(path: Path, mask: np.ndarray) -> None:
    save_binary_mask(path, mask)


def _minimal_result_payload(take_id: str, artifacts: list[dict]) -> dict:
    return {
        "take_id": take_id,
        "status": "ok",
        "processed_at": "2026-05-29T22:43:12+00:00",
        "processing_pipeline": {
            "id": "mining_steel_ball_classification_25d",
            "pipeline_family": "25d",
        },
        "artifacts": artifacts,
        "files": {item["artifact_id"]: item["path"] for item in artifacts},
        "stage_params": {
            "detect_belt_plane": {"gradient_threshold_value": 2.5},
            "remove_belt_segment_objects": {"min_height_mm": 8.0},
            "known_object_25d": {},
        },
    }


def test_mask_helpers_compute_expected_metrics(tmp_path: Path) -> None:
    before = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    after = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    diff = mask_a_minus_b(before, after)
    assert np.array_equal(diff, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool))
    assert np.array_equal(mask_intersection(before, after), np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool))
    assert np.array_equal(mask_union(before, after), np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=bool))

    metrics = compute_support_loss(before, after, step="test")
    assert metrics["before_pixels"] == 3
    assert metrics["after_pixels"] == 2
    assert metrics["removed_pixels"] == 2
    assert metrics["added_pixels"] == 1
    assert metrics["kept_pixels"] == 1
    assert metrics["removed_fraction"] == pytest.approx(2 / 3)
    assert metrics["iou"] == pytest.approx(1 / 4)

    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, before)
    loaded = load_binary_mask(mask_path)
    assert loaded is not None
    assert np.array_equal(loaded, before)

    overlay = overlay_diff(before, after)
    assert overlay.shape == (3, 3, 3)


def test_build_bundle_from_minimal_result_records_missing_without_failing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "audit_take_minimal"
    processed = data_dir / "processed" / take_id
    processed.mkdir(parents=True)

    valid = np.zeros((8, 8), dtype=bool)
    valid[1:7, 1:7] = True
    low_grad = valid.copy()
    low_grad[2:6, 2:6] = False

    valid_path = processed / "valid_mask.png"
    low_grad_path = processed / "low_gradient_mask.png"
    _write_mask(valid_path, valid)
    _write_mask(low_grad_path, low_grad)

    artifacts = [
        {"artifact_id": "valid_mask", "path": "valid_mask.png", "stage_id": "detect_belt_plane", "kind": "image"},
        {"artifact_id": "low_gradient_mask", "path": "low_gradient_mask.png", "stage_id": "detect_belt_plane", "kind": "image"},
    ]
    payload = _minimal_result_payload(take_id, artifacts)
    (processed / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (processed / "gradient_debug.json").write_text(
        json.dumps({"gradient_threshold": 1.5, "height_gate": {"status": "applied", "coverage_ratio": 0.5}}),
        encoding="utf-8",
    )

    bundle = build_reference_audit_bundle(
        data_dir,
        take_id=take_id,
        output_root=data_dir / "debug" / "25d_reference_audit",
        include_overlays=False,
    )

    assert (bundle.output_dir / "manifest.json").is_file()
    assert (bundle.output_dir / "report.md").is_file()
    assert (bundle.output_dir / "index.html").is_file()
    assert (bundle.output_dir / "stage_params_resolved.json").is_file()
    assert "valid_mask" in bundle.manifest["artifacts_found"]
    assert "reference_surface_selected_mask" in bundle.manifest["artifacts_missing"]
    assert bundle.artifacts_copied >= 3
    assert (bundle.output_dir / "10_summary" / "support_loss.json").is_file()
    assert (bundle.output_dir / "10_summary" / "support_loss.csv").is_file()


def test_build_bundle_from_synthetic_run(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="audit_synthetic")
    run_ball_inspection_25d_flow(data_dir, take_id=take_id)

    bundle = build_reference_audit_bundle(
        data_dir,
        take_id=take_id,
        output_root=data_dir / "debug" / "25d_reference_audit",
    )
    assert bundle.output_dir.is_dir()
    assert bundle.manifest["pipeline_id"] == "mining_steel_ball_classification_25d"
    assert len(bundle.manifest["artifacts_found"]) > 5
    assert (bundle.output_dir / "06_candidate_support" / "reference_surface_selected_mask.png").is_file() or (
        "reference_surface_selected_mask" in bundle.manifest["artifacts_missing"]
    )


def test_build_bundle_includes_blob_cluster_artifacts_when_available(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="audit_blob")
    run_ball_inspection_25d_flow(
        data_dir,
        take_id=take_id,
        stage_params={"detect_belt_plane": {"background_detection_strategy": "low_gradient_blob_height_clusters"}},
    )

    bundle = build_reference_audit_bundle(
        data_dir,
        take_id=take_id,
        output_root=data_dir / "debug" / "25d_reference_audit",
    )
    assert (bundle.output_dir / "04_blob_clusters" / "selected_blob_cluster_mask.png").is_file()
    assert (bundle.output_dir / "04_blob_clusters" / "blob_cluster_selection_debug.json").is_file()
    assert (bundle.output_dir / "04_blob_clusters" / "height_border_strength.png").is_file()
    assert (bundle.output_dir / "04_blob_clusters" / "height_border_split_debug.json").is_file()
    assert (bundle.output_dir / "04_blob_clusters" / "height_split_blob_fragments_overlay.png").is_file()
    assert (bundle.output_dir / "04_blob_clusters" / "height_split_debug.json").is_file()
    support_loss = json.loads((bundle.output_dir / "10_summary" / "support_loss.json").read_text(encoding="utf-8"))
    steps = {row.get("step") for row in support_loss if isinstance(row, dict)}
    assert "rejected_by_blob_cluster_selection" in steps


def test_resolve_processed_run_latest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "audit_take_latest"
    processed = data_dir / "processed" / take_id
    processed.mkdir(parents=True)
    payload = _minimal_result_payload(take_id, [])
    (processed / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_processed_run(data_dir, take_id=take_id, run_id="latest")
    assert resolved is not None
    assert resolved.take_id == take_id
    assert resolved.result_path == processed / "result.json"


def test_build_bundle_tolerates_list_debug_json_payloads(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "audit_take_list_payload"
    processed = data_dir / "processed" / take_id
    processed.mkdir(parents=True)

    valid = np.ones((8, 8), dtype=bool)
    _write_mask(processed / "valid_mask.png", valid)

    artifacts = [
        {"artifact_id": "valid_mask", "path": "valid_mask.png", "stage_id": "detect_belt_plane", "kind": "image"},
        {
            "artifact_id": "reference_surface_plateaus",
            "path": "reference_surface_plateaus.json",
            "stage_id": "detect_belt_plane",
            "kind": "json",
        },
    ]
    payload = _minimal_result_payload(take_id, artifacts)
    (processed / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (processed / "reference_surface_plateaus.json").write_text(json.dumps([{"z_min_mm": 1.0, "z_max_mm": 2.0}]), encoding="utf-8")
    (processed / "gradient_debug.json").write_text(
        json.dumps({"gradient_threshold": 1.5, "height_gate": {"status": "applied"}}),
        encoding="utf-8",
    )

    bundle = build_reference_audit_bundle(
        data_dir,
        take_id=take_id,
        output_root=data_dir / "debug" / "25d_reference_audit",
        include_overlays=False,
        include_diffs=False,
    )

    assert (bundle.output_dir / "manifest.json").is_file()
    assert (bundle.output_dir / "report.md").is_file()
    assert any(
        "reference_surface_plateaus.json payload was list" in warning for warning in bundle.manifest["warnings"]
    )
    report = (bundle.output_dir / "report.md").read_text(encoding="utf-8")
    assert "reference_surface_plateaus.json payload was list" in report
    assert bundle.manifest["derived_thresholds"]["gradient_threshold"] == 1.5
    assert "selected_plateau_z_band" not in bundle.manifest["derived_thresholds"]


def test_extract_derived_thresholds_from_dict_payloads(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "gradient_debug.json").write_text(
        json.dumps({"gradient_threshold": 2.1, "height_gate": {"status": "applied", "coverage_ratio": 0.4}}),
        encoding="utf-8",
    )
    (processed / "reference_surface_plateaus.json").write_text(
        json.dumps({"selected_plateau": {"z_band_mm": [10.0, 12.5]}}),
        encoding="utf-8",
    )
    (processed / "plane_fit_debug.json").write_text(
        json.dumps({"effective_ransac_threshold_mm": 1.8, "residual_tolerance_mm": 2.2}),
        encoding="utf-8",
    )

    run = ResolvedRun(
        take_id="t1",
        pipeline_id="mining_steel_ball_classification_25d",
        run_id="processed",
        run_descriptor="latest",
        processed_dir=processed,
        result_path=processed / "result.json",
        result_payload=_minimal_result_payload("t1", []),
    )
    thresholds, warnings = _extract_derived_thresholds(processed, {}, run)

    assert warnings == []
    assert thresholds["gradient_threshold"] == 2.1
    assert thresholds["height_gate"] == {"status": "applied", "coverage_ratio": 0.4}
    assert thresholds["selected_plateau_z_band"] == [10.0, 12.5]
    assert thresholds["ransac_threshold_mm"] == 1.8
    assert thresholds["expansion_tolerance_mm"] == 2.2


def test_cli_debug_reference_accepts_take_id(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    take_id = "audit_cli_take"
    processed = data_dir / "processed" / take_id
    processed.mkdir(parents=True)
    valid = np.ones((4, 4), dtype=bool)
    _write_mask(processed / "valid_mask.png", valid)
    payload = _minimal_result_payload(
        take_id,
        [{"artifact_id": "valid_mask", "path": "valid_mask.png", "stage_id": "detect_belt_plane", "kind": "image"}],
    )
    (processed / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "25d",
            "debug-reference",
            "--data-dir",
            str(data_dir),
            "--take-id",
            take_id,
            "--output-dir",
            str(data_dir / "debug" / "25d_reference_audit"),
            "--no-include-overlays",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "25D reference audit bundle generated" in out
    assert (data_dir / "debug" / "25d_reference_audit" / take_id).is_dir()


def test_cli_debug_reference_without_result_exits_helpfully(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "25d",
            "debug-reference",
            "--data-dir",
            str(data_dir),
            "--take-id",
            "missing_take",
        ],
    )
    code = cli.main()
    err = capsys.readouterr().err
    assert code == 2
    assert "No processed result found" in err


def test_cli_debug_reference_rerun_processes_then_bundles(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    take_id, _ = create_synthetic_25d_take(data_dir, session_id="audit_rerun")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "25d",
            "debug-reference",
            "--data-dir",
            str(data_dir),
            "--take-id",
            take_id,
            "--rerun",
            "--output-dir",
            str(data_dir / "debug" / "25d_reference_audit"),
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Re-running 25D pipeline before audit bundle generation" in out
    assert "25D reference audit bundle generated" in out
    assert (data_dir / "processed" / take_id / "result.json").is_file()
