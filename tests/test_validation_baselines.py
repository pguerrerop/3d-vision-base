from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.validation import ValidationService
from vision_3d_acquisition.validation.service import COMPARATOR_BY_SEMANTIC_TYPE


def _settings(root: Path) -> ApiSettings:
    return ApiSettings(root, root / "incoming", root / "processed", root / "state", root / "events", root / "sessions", root / "datasets")


def _resolved(root: Path, artifact: Path) -> dict:
    return {"artifact_root": root, "artifacts": [{"artifact_id": "final_object_mask", "stage_id": "segment", "processing_unit_id": "segment.final", "kind": "mask", "path": artifact.name}], "result_payload": {"processing_pipeline": {"processing_units": []}, "recipe_snapshot": {}}, "stage_params": {}, "recipe_id": None}


def test_snapshot_reapproval_integrity_and_rebuild(tmp_path: Path) -> None:
    source = tmp_path / "candidate.png"
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(source)
    store = ValidationService(_settings(tmp_path))
    store._resolve = lambda **_: _resolved(tmp_path, source)  # type: ignore[method-assign]
    request = {"take_id": "take_1", "pipeline_id": "mining_steel_ball_classification_25d", "run_id": "run_1", "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask"}
    first = store.create_baselines(request)["baselines"][0]
    snapshot = tmp_path / "validation" / "baselines" / first["id"] / first["snapshot_artifact_path"]
    assert snapshot.is_file() and store.verify_integrity()["ok"]
    source.unlink()
    assert snapshot.is_file(), "baseline must outlive source run cleanup"
    source.write_bytes(snapshot.read_bytes())
    second = store.create_baselines(request)["baselines"][0]
    assert second["version"] == 2 and store.get_baseline(first["id"])["status"] == "inactive"
    assert store.get_baseline(second["id"])["status"] == "active"
    (tmp_path / "validation" / "indexes" / "baselines.json").unlink()
    assert store.rebuild_indexes()["baselines"] == 2


def test_binary_mask_semantics_and_unknown_artifacts(tmp_path: Path) -> None:
    store = ValidationService(_settings(tmp_path))
    left, right = tmp_path / "left.png", tmp_path / "right.png"
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(left)
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(right)
    base = {"id": "baseline_x", "comparison_policy": {"comparator": "binary_mask"}, "artifact_semantic_type": "binary_mask", "version": 1}
    result = store._mask(left, right, base, {"artifact_id": "mask"})
    assert result["status"] == "pass" and result["metrics"]["iou"] == 1.0
    Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(right)
    result = store._mask(left, right, base, {"artifact_id": "mask"}, tmp_path / "validation" / "executions" / "e" / "diffs")
    assert result["status"] == "regression"
    assert result["metrics"]["baseline_components"] == 1
    assert len(result["diff_artifacts"]) == 5
    assert (tmp_path / "validation" / result["diff_artifacts"][-1]).is_file()


def test_numeric_raster_persists_heatmap_and_checks_shape(tmp_path: Path) -> None:
    store = ValidationService(_settings(tmp_path)); left, right = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(left, np.array([[1.0, 2.0]], dtype=np.float32)); np.save(right, np.array([[1.0, 3.0]], dtype=np.float32))
    base = {"id":"b", "version":1, "artifact_semantic_type":"numeric_raster", "comparison_policy":{"max_mae":0.1}}
    result = store._raster(left, right, base, {"artifact_id":"height"}, tmp_path / "validation" / "executions" / "e" / "diffs")
    assert result["status"] == "regression" and result["metrics"]["mae"] == 0.5
    assert len(result["diff_artifacts"]) == 2
    np.save(right, np.array([[1.0]], dtype=np.float32))
    assert store._raster(left, right, base, {"artifact_id":"height"})["status"] == "not_comparable"


def test_measurement_and_classification_semantics(tmp_path: Path) -> None:
    store = ValidationService(_settings(tmp_path)); left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps({"objects":[{"object_id":"one","diameter_mm":10.0,"class_name":"sphere","superclass":"ball","confidence":.9}]}))
    right.write_text(json.dumps({"objects":[{"object_id":"one","diameter_mm":11.0,"class_name":"fragment","superclass":"fragment","confidence":.8,"warnings":["invalid_geometry"]}]}))
    measurements = store._measurement_compare(left, right, {"id":"m","version":1,"comparison_policy":{"metrics":{"diameter_selected_mm":{"absolute_tolerance":.5,"required":True,"units":"mm"}}}}, {"artifact_id":"measure"})
    assert measurements["status"] == "regression"
    metric = measurements["object_results"][0]["metric_results"][0]
    assert metric["effective_tolerance"] == .5 and metric["reason"] == "tolerance_exceeded"
    classification = store._classification_compare(left, right, {"id":"c","version":1,"comparison_policy":{"critical_warning_codes":["invalid_geometry"],"max_confidence_drop":.15}}, {"artifact_id":"classify"})
    assert classification["status"] == "regression"
    assert {"superclass_changed","label_changed","new_critical_warning"}.issubset(classification["object_results"][0]["reasons"])


def test_archived_suite_rejects_case_mutation(tmp_path: Path) -> None:
    source = tmp_path / "candidate.png"
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(source)
    store = ValidationService(_settings(tmp_path))
    store._resolve = lambda **_: _resolved(tmp_path, source)  # type: ignore[method-assign]
    pipeline = "mining_steel_ball_classification_25d"
    baseline = store.create_baselines({"take_id": "take_1", "pipeline_id": pipeline, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask"})["baselines"][0]
    suite = store.create_suite({"name": "regression", "pipeline_id": pipeline})
    case = store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})
    store.archive_suite(suite["id"])

    for mutation in (
        lambda: store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]}),
        lambda: store.update_case(suite["id"], case["id"], {"enabled": False}),
        lambda: store.delete_case(suite["id"], case["id"]),
    ):
        with pytest.raises(ValueError, match="archived"):
            mutation()
    assert [x["id"] for x in store.get_suite(suite["id"])["cases"]] == [case["id"]]

    # Archived suites stay inspectable, and duplication targets the active copy.
    duplicate = store.duplicate_suite(suite["id"])
    assert duplicate["status"] == "active" and len(duplicate["cases"]) == 1

    store.restore_suite(suite["id"])
    assert store.delete_case(suite["id"], case["id"])["deleted"] == case["id"]
    assert store.get_suite(suite["id"])["cases"] == []
    # Removing a case drops suite membership only.
    assert store.get_baseline(baseline["id"])["status"] == "active"
    assert store.verify_integrity()["ok"]


def test_comparator_resolution_prefers_declared_contract_over_identifier(tmp_path: Path) -> None:
    store = ValidationService(_settings(tmp_path))

    # A stage stating the comparator outright wins over every weaker signal.
    stated = store._classify({"artifact_id": "belt_stripes_mask", "kind": "image", "metadata": {"comparator": "numeric_raster", "semantic_type": "belt_stripes_mask"}})
    assert stated == {"comparator": "numeric_raster", "source": "declared_comparator"}

    # A declared semantic type resolves through the reviewed table.
    declared = store._classify({"artifact_id": "opaque_name", "kind": "image", "metadata": {"semantic_type": "plane_residual_heatmap"}})
    assert declared == {"comparator": "numeric_raster", "source": "semantic_type"}
    assert COMPARATOR_BY_SEMANTIC_TYPE["valid_mask"] == "binary_mask"

    # Component id maps are label rasters, so they are deliberately unmapped and
    # must not be silently treated as binary masks by the table.
    assert "blob_components_id_mask" not in COMPARATOR_BY_SEMANTIC_TYPE

    # A declared role can settle comparability on its own.
    assert store._classify({"artifact_id": "x", "kind": "image", "metadata": {"role": "difference_overlay"}}) == {"comparator": "visual_only", "source": "role"}

    # The on-disk representation is authoritative for numeric rasters.  The old
    # identifier rules could never reach this: "npy" is not a valid artifact kind.
    assert store._classify({"artifact_id": "belt_residual", "kind": "image", "path": "run/belt_residual.npy"}) == {"comparator": "numeric_raster", "source": "path"}

    # Identifier heuristics still resolve, but report themselves as guesses.
    guessed = store._classify({"artifact_id": "final_object_mask", "kind": "image"})
    assert guessed == {"comparator": "binary_mask", "source": "heuristic"}

    # Nothing to go on stays explicitly unresolved.
    assert store._classify({"artifact_id": "mystery", "kind": "file"}) == {"comparator": "not_comparable", "source": "unresolved"}


def test_declared_contract_is_recorded_and_survives_a_policy_without_comparator(tmp_path: Path) -> None:
    source = tmp_path / "candidate.png"
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(source)
    store = ValidationService(_settings(tmp_path))
    resolved = _resolved(tmp_path, source)
    resolved["artifacts"][0]["metadata"] = {"semantic_type": "belt_background_mask", "role": "fit_support"}
    store._resolve = lambda **_: resolved  # type: ignore[method-assign]

    # A policy that tunes thresholds without naming a comparator is realistic, and
    # used to be the case where the semantic-type fallback carried the decision.
    baseline = store.create_baselines({
        "take_id": "take_1", "pipeline_id": "mining_steel_ball_classification_25d", "run_id": "run_1",
        "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask",
        "comparison_policy": {"min_iou": 0.99},
    })["baselines"][0]

    assert baseline["comparator"] == "binary_mask"
    assert baseline["comparator_source"] == "semantic_type"
    assert baseline["artifact_semantic_type"] == "belt_background_mask"
    assert baseline["artifact_role"] == "fit_support"
    # The declared type is not a comparator name, so resolution must not fall back to it.
    assert store._comparator_of(baseline) == "binary_mask"

    result = store.compare(baseline["id"], candidate_run_id="run_1")
    assert result["status"] == "pass" and result["comparator"] == "binary_mask"


def test_unresolved_artifacts_are_skipped_with_their_classification_source(tmp_path: Path) -> None:
    source = tmp_path / "candidate.bin"
    source.write_bytes(b"opaque")
    store = ValidationService(_settings(tmp_path))
    resolved = _resolved(tmp_path, source)
    resolved["artifacts"][0].update({"artifact_id": "mystery_blob", "kind": "file", "path": source.name, "metadata": {}})
    store._resolve = lambda **_: resolved  # type: ignore[method-assign]

    outcome = store.create_baselines({
        "take_id": "take_1", "pipeline_id": "mining_steel_ball_classification_25d", "run_id": "run_1",
        "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "mystery_blob",
    })
    assert outcome["baselines"] == []
    assert outcome["skipped"] == [{"artifact_id": "mystery_blob", "reason": "unsupported", "comparator_source": "unresolved"}]
