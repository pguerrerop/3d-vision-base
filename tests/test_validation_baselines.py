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


# --------------------------------------------------------------------------
# Baseline resolution: active / pinned / allowed_versions
# --------------------------------------------------------------------------

PIPELINE = "mining_steel_ball_classification_25d"
_V1 = np.array([[0, 255], [0, 0]], dtype=np.uint8)
_V2 = np.array([[255, 255], [0, 0]], dtype=np.uint8)


def _resolution_fixture(tmp_path: Path) -> tuple[ValidationService, Path, dict, dict]:
    """Two approved versions of one mask, with the candidate run holding v2."""
    source = tmp_path / "candidate.png"
    Image.fromarray(_V1).save(source)
    store = ValidationService(_settings(tmp_path))
    store._resolve = lambda **_: _resolved(tmp_path, source)  # type: ignore[method-assign]
    request = {"take_id": "take_1", "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask"}
    v1 = store.create_baselines(request)["baselines"][0]
    Image.fromarray(_V2).save(source)
    v2 = store.create_baselines(request)["baselines"][0]
    assert (v1["version"], v2["version"]) == (1, 2)
    assert store.get_baseline(v1["id"])["status"] == "inactive"
    return store, source, v1, v2


def _case_of(store: ValidationService, suite_id: str, case_id: str) -> dict:
    return next(x for x in store.get_suite(suite_id)["cases"] if x["id"] == case_id)


def test_active_resolution_follows_the_activated_version(tmp_path: Path) -> None:
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})

    # A case configured against v1 in active mode resolves to whatever is active.
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})
    assert case["baseline_resolution"] == {"mode": "active"}
    assert store._case_baselines(case) == [v2["id"]]

    # A legacy case carries no resolution at all and must behave as active.
    legacy = dict(case); legacy.pop("baseline_resolution")
    assert store._case_baselines(legacy) == [v2["id"]]

    # Reactivating v1 moves active resolution back to the highest active version.
    store.set_active(v2["id"], False)
    store.set_active(v1["id"], True)
    assert store._case_baselines(case) == [v1["id"]]


def test_pinned_resolution_never_falls_back(tmp_path: Path) -> None:
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": v1["id"]}})

    # v1 is already inactive here: pinning holds regardless of activation state.
    assert store.get_baseline(v1["id"])["status"] == "inactive"
    assert store._case_baselines(case) == [v1["id"]]

    # Activating a newer version must not move the pin.
    store.set_active(v2["id"], True)
    assert store._case_baselines(_case_of(store, suite["id"], case["id"])) == [v1["id"]]

    # Comparison still runs against the pinned snapshot, so the candidate (v2
    # content) diverges from it rather than silently passing against active.
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    assert execution["cases"][0]["status"] == "regression"
    assert execution["cases"][0]["comparisons"][0]["baseline_ref"]["baseline_id"] == v1["id"]


def test_missing_pinned_baseline_stays_explicit(tmp_path: Path) -> None:
    store, _, v1, _ = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": v1["id"]}})

    # Simulate a snapshot that is no longer readable as a record.
    (tmp_path / "validation" / "baselines" / v1["id"] / "baseline.json").unlink()
    assert store._case_baselines(case) == [v1["id"]], "the configured pin survives a missing record"

    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    comparison = execution["cases"][0]["comparisons"][0]
    assert comparison["status"] == "not_comparable"
    assert v1["id"] in comparison["summary"]


def test_allowed_versions_passes_on_any_configured_match_and_persists_it(tmp_path: Path) -> None:
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], v2["id"]]}})
    assert store._case_baselines(case) == [v1["id"], v2["id"]]

    # The candidate matches v2 and diverges from v1: one match is enough.
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    result = execution["cases"][0]
    assert result["status"] == "pass"
    assert result["matched_baseline_id"] == v2["id"], "the matched version is persisted on the execution"
    assert {c["status"] for c in result["comparisons"]} == {"pass", "regression"}


def test_allowed_versions_rejects_duplicates_and_incompatible_families(tmp_path: Path) -> None:
    store, source, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})

    with pytest.raises(ValueError, match="unique baseline IDs"):
        store.add_case(suite["id"], {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], v1["id"]]}})

    with pytest.raises(ValueError, match="invalid baseline resolution mode"):
        store.add_case(suite["id"], {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "whatever"}})

    # A baseline from another take must not be accepted into a case.
    other = store.create_baselines({"take_id": "take_2", "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask"})["baselines"][0]
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], v2["id"]]}})
    with pytest.raises(ValueError, match="incompatible"):
        store.update_case(suite["id"], case["id"], {"baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], other["id"]]}})

    # A suite whose pipeline differs from the baseline is refused outright.
    foreign = store.create_suite({"name": "other", "pipeline_id": "some_other_pipeline"})
    with pytest.raises(ValueError, match="pipeline does not match"):
        store.add_case(foreign["id"], {"baseline_ids": [v1["id"]]})


def test_disabled_cases_are_skipped_without_comparisons(tmp_path: Path) -> None:
    store, _, v1, _ = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]], "enabled": False})

    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    result = execution["cases"][0]
    assert result["skipped"] is True and "comparisons" not in result
    assert result["case_id"] == case["id"]

    # Re-enabling brings the case back into the next execution.
    store.update_case(suite["id"], case["id"], {"enabled": True})
    assert "comparisons" in store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})["cases"][0]


def test_activating_a_baseline_does_not_rewrite_stored_executions(tmp_path: Path) -> None:
    """The load-bearing invariant: history reflects the configuration of its own moment."""
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})

    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    stored = json.loads((tmp_path / "validation" / "executions" / execution["id"] / "execution.json").read_text())
    assert stored["cases"][0]["comparisons"][0]["baseline_ref"]["baseline_id"] == v2["id"]

    # Change present governance in every way that could plausibly leak backwards.
    store.set_active(v2["id"], False)
    store.set_active(v1["id"], True)
    store.update_case(suite["id"], case["id"], {"baseline_resolution": {"mode": "pinned", "baseline_id": v1["id"]}, "enabled": False, "tags": ["moved"]})

    assert store.get_execution(execution["id"]) == stored
    assert json.loads((tmp_path / "validation" / "executions" / execution["id"] / "execution.json").read_text()) == stored


def test_an_explicitly_empty_selection_is_rejected_on_both_mutation_paths(tmp_path: Path) -> None:
    """An absent selection may default; a cleared one is a request to correct."""
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})

    for payload in ({"mode": "allowed_versions", "allowed_baseline_ids": []}, {"mode": "pinned", "baseline_id": None}):
        with pytest.raises(ValueError, match="requires"):
            store.add_case(suite["id"], {"baseline_ids": [v1["id"]], "baseline_resolution": payload})

    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"], v2["id"]]})
    with pytest.raises(ValueError, match="at least one baseline version"):
        store.update_case(suite["id"], case["id"], {"baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": []}})
    assert _case_of(store, suite["id"], case["id"])["baseline_resolution"] == {"mode": "active"}, "a rejected edit leaves the case untouched"

    # Omitting the key entirely still defaults to the configured baselines.
    defaulted = store.add_case(suite["id"], {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "allowed_versions"}})
    assert defaulted["baseline_resolution"]["allowed_baseline_ids"] == [v1["id"], v2["id"]]
    pinned = store.add_case(suite["id"], {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "pinned"}})
    assert pinned["baseline_resolution"]["baseline_id"] == v1["id"]


def test_add_case_validates_the_pin_like_update_case_does(tmp_path: Path) -> None:
    store, _, v1, _ = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})

    with pytest.raises(ValueError, match="incompatible"):
        store.add_case(suite["id"], {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": "baseline_does_not_exist"}})

    other_take = store.create_baselines({"take_id": "take_2", "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask"})["baselines"][0]
    with pytest.raises(ValueError, match="incompatible"):
        store.add_case(suite["id"], {"take_id": "take_1", "baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": other_take["id"]}})
    with pytest.raises(ValueError, match="incompatible"):
        store.add_case(suite["id"], {"take_id": "take_1", "baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], other_take["id"]]}})

    assert store.get_suite(suite["id"])["cases"] == [], "no rejected case was persisted"


# --------------------------------------------------------------------------
# Promotion: candidate -> new baseline
# --------------------------------------------------------------------------

def _promotion_fixture(tmp_path: Path) -> tuple[ValidationService, Path, dict, dict, dict]:
    """One approved baseline, then a candidate run whose output diverges from it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "candidate.png"
    Image.fromarray(_V1).save(source)
    store = ValidationService(_settings(tmp_path))
    store._resolve = lambda **_: _resolved(tmp_path, source)  # type: ignore[method-assign]
    v1 = store.create_baselines({"take_id": "take_1", "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "segment", "artifact_id": "final_object_mask", "comparison_policy": {"comparator": "binary_mask", "min_iou": 0.99}})["baselines"][0]

    Image.fromarray(_V2).save(source)  # the candidate now disagrees with the baseline
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    assert execution["cases"][0]["status"] == "regression"
    return store, source, v1, case, execution


def test_promotion_requires_exact_execution_case_and_comparison_context(tmp_path: Path) -> None:
    store, _, v1, case, execution = _promotion_fixture(tmp_path)

    with pytest.raises(KeyError):
        store.promote_comparison("execution_nope", case["id"], v1["id"], {})
    with pytest.raises(KeyError):
        store.promote_comparison(execution["id"], "case_nope", v1["id"], {})
    with pytest.raises(KeyError):
        store.promote_comparison(execution["id"], case["id"], "baseline_nope", {})
    assert len(store.baseline_history(take_id="take_1")) == 1, "no failed lookup created a version"


def test_promotion_creates_the_next_version_without_touching_the_previous_one(tmp_path: Path) -> None:
    store, _, v1, case, execution = _promotion_fixture(tmp_path)
    before_record = store.get_baseline(v1["id"])
    before_snapshot = (tmp_path / "validation" / "baselines" / v1["id"] / v1["snapshot_artifact_path"]).read_bytes()

    outcome = store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True, "reviewed_by": "pablo", "notes": "accepted new segmentation"})
    promoted = outcome["baseline"]

    assert outcome["already_promoted"] is False
    assert promoted["version"] == 2 and promoted["status"] == "active"
    assert promoted["supersedes_baseline_id"] == v1["id"]

    # The promoted snapshot is the candidate output, not a copy of the old baseline.
    promoted_snapshot = (tmp_path / "validation" / "baselines" / promoted["id"] / promoted["snapshot_artifact_path"]).read_bytes()
    assert promoted_snapshot != before_snapshot
    assert promoted["artifact_checksum"] == promoted["promotion"]["candidate_artifact_checksum"]

    # Provenance is complete and points back at the exact context.
    assert promoted["promotion"]["source_execution_id"] == execution["id"]
    assert promoted["promotion"]["source_case_id"] == case["id"]
    assert promoted["promotion"]["source_comparison_id"] == v1["id"]
    assert promoted["promotion"]["candidate_run_id"] == "run_1"
    assert promoted["promotion"]["promoted_by"] == "pablo"
    assert promoted["promotion"]["notes"] == "accepted new segmentation"

    # v1's own record keeps every field except the activation flag the new version flipped.
    after_record = store.get_baseline(v1["id"])
    assert after_record["status"] == "inactive"
    assert {k: v for k, v in after_record.items() if k != "status"} == {k: v for k, v in before_record.items() if k != "status"}
    assert (tmp_path / "validation" / "baselines" / v1["id"] / v1["snapshot_artifact_path"]).read_bytes() == before_snapshot
    assert store.verify_integrity()["ok"]


def test_promotion_is_inactive_by_default_and_activation_is_a_separate_decision(tmp_path: Path) -> None:
    store, _, v1, case, execution = _promotion_fixture(tmp_path)

    outcome = store.promote_comparison(execution["id"], case["id"], v1["id"], {})
    promoted = outcome["baseline"]
    assert promoted["status"] == "inactive", "promotion does not activate unless asked"
    assert store.get_baseline(v1["id"])["status"] == "active", "the previous active version stays active"
    assert outcome["resulting_active_baseline"]["id"] == v1["id"]

    # An active-mode case therefore keeps resolving to v1 until someone activates v2.
    assert store._case_baselines(_case_of(store, execution["suite_id"], case["id"])) == [v1["id"]]
    store.set_active(promoted["id"], True)
    store.set_active(v1["id"], False)
    assert store._case_baselines(_case_of(store, execution["suite_id"], case["id"])) == [promoted["id"]]


def test_repeated_promotion_of_the_same_candidate_reuses_the_existing_baseline(tmp_path: Path) -> None:
    store, source, v1, case, execution = _promotion_fixture(tmp_path)
    first = store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True})

    second = store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True})
    assert second["already_promoted"] is True
    assert second["baseline"]["id"] == first["baseline"]["id"]
    assert second["baseline"]["version"] == 2
    assert len(store.baseline_history(take_id="take_1")) == 2, "no third version was created"

    # Idempotency is keyed on the candidate content, so a changed candidate is a
    # genuinely new promotion rather than a silent reuse.
    Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(source)
    third = store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True})
    assert third["already_promoted"] is False and third["baseline"]["version"] == 3


def test_policy_carry_forward_is_explicit(tmp_path: Path) -> None:
    store, _, v1, case, execution = _promotion_fixture(tmp_path)
    assert v1["comparison_policy"] == {"comparator": "binary_mask", "min_iou": 0.99}

    carried = store.promote_comparison(execution["id"], case["id"], v1["id"], {})
    assert carried["baseline"]["comparison_policy"] == v1["comparison_policy"]

    store2, _, w1, case2, execution2 = _promotion_fixture(tmp_path / "reset")
    reset = store2.promote_comparison(execution2["id"], case2["id"], w1["id"], {"carry_forward_policy": False})
    assert reset["baseline"]["comparison_policy"] != w1["comparison_policy"]


def test_promotion_impact_separates_cases_that_follow_from_cases_that_do_not(tmp_path: Path) -> None:
    store, _, v1, active_case, execution = _promotion_fixture(tmp_path)
    suite_id = execution["suite_id"]

    pinned = store.add_case(suite_id, {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": v1["id"]}})
    allowed = store.add_case(suite_id, {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"]]}})
    disabled = store.add_case(suite_id, {"baseline_ids": [v1["id"]], "enabled": False})
    archived_suite = store.create_suite({"name": "old", "pipeline_id": PIPELINE})
    archived_case = store.add_case(archived_suite["id"], {"baseline_ids": [v1["id"]]})
    store.archive_suite(archived_suite["id"])

    impact = store.promote_comparison(execution["id"], active_case["id"], v1["id"], {"activate": True})["affected_case_resolution"]
    assert [x["case_id"] for x in impact["active_cases"]] == [active_case["id"]]
    assert [x["case_id"] for x in impact["pinned_cases_unchanged"]] == [pinned["id"]]
    assert [x["case_id"] for x in impact["allowed_version_cases_unchanged"]] == [allowed["id"]]
    assert [x["case_id"] for x in impact["disabled_cases"]] == [disabled["id"]]
    assert [x["case_id"] for x in impact["archived_suites"]] == [archived_case["id"]]


def test_promotion_reports_a_missing_candidate_artifact(tmp_path: Path) -> None:
    store, source, v1, case, execution = _promotion_fixture(tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match="candidate artifact missing"):
        store.promote_comparison(execution["id"], case["id"], v1["id"], {})
    assert len(store.baseline_history(take_id="take_1")) == 1


def test_promotion_reports_state_after_the_promotion_not_before(tmp_path: Path) -> None:
    """The impact summary drives a review screen, so it must not describe stale state."""
    store, _, v1, case, execution = _promotion_fixture(tmp_path)
    outcome = store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True})

    previous = outcome["previous_active_baseline"]
    assert previous["id"] == v1["id"]
    assert previous["status"] == "inactive", "activating v2 deactivated v1, and the response must say so"
    assert previous["status"] == store.get_baseline(v1["id"])["status"]

    resulting = outcome["resulting_active_baseline"]
    assert resulting["id"] == outcome["baseline"]["id"] and resulting["status"] == "active"

    # The idempotent retry describes the same post-promotion state.
    again = store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True})
    assert again["already_promoted"] is True
    assert again["previous_active_baseline"]["status"] == "inactive"
    assert again["resulting_active_baseline"]["id"] == resulting["id"]


def test_promotion_does_not_claim_an_active_baseline_when_none_is_active(tmp_path: Path) -> None:
    store, _, v1, case, execution = _promotion_fixture(tmp_path)
    store.set_active(v1["id"], False)  # the family now has no active version at all

    outcome = store.promote_comparison(execution["id"], case["id"], v1["id"], {})
    assert outcome["baseline"]["status"] == "inactive"
    assert outcome["resulting_active_baseline"] is None, "reporting an inactive baseline as the active one is a lie"
    assert [b["id"] for b in store.baseline_history(take_id="take_1") if b["status"] == "active"] == []

    # Activating the promoted version is what makes the family resolvable again.
    store.set_active(outcome["baseline"]["id"], True)
    assert store._active_of_family(outcome["baseline"])["id"] == outcome["baseline"]["id"]


def test_promotion_refuses_a_candidate_that_changes_underneath_it(tmp_path: Path) -> None:
    """The recorded provenance checksum must describe the snapshot that was stored."""
    store, source, v1, case, execution = _promotion_fixture(tmp_path)
    real_create = store.create_baselines

    def _create_with_a_racing_write(request):
        Image.fromarray(np.array([[7, 7], [7, 7]], dtype=np.uint8)).save(source)
        return real_create(request)

    store.create_baselines = _create_with_a_racing_write  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="changed while it was being promoted"):
        store.promote_comparison(execution["id"], case["id"], v1["id"], {"activate": True})


# --------------------------------------------------------------------------
# HTTP route layer: the contract the typed frontend bindings assume
# --------------------------------------------------------------------------

def test_case_routes_accept_and_return_what_the_client_bindings_declare(tmp_path: Path) -> None:
    """Payload and response shapes here mirror api/client.ts exactly.

    A binding that typechecks can still send the wrong shape, so these assertions
    pin the fields the frontend types promise.
    """
    from vision_3d_acquisition.api import validation as routes

    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})

    # addValidationCase(suiteId, ValidationCaseRequest)
    created = routes.add_case(
        suite["id"],
        {
            "take_id": "take_1",
            "baseline_ids": [v1["id"], v2["id"]],
            "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], v2["id"]]},
            "enabled": True,
            "tags": ["nightly"],
            "notes": "regression watch",
            "included_artifacts": [],
        },
        store=store,
    )
    for field in ("id", "take_id", "baseline_ids", "baseline_resolution", "enabled", "tags", "notes"):
        assert field in created, f"ValidationSuiteCase declares {field}"
    assert created["baseline_resolution"]["mode"] == "allowed_versions"

    # updateValidationCase(suiteId, caseId, ValidationCaseUpdate) — a partial payload,
    # and the fields it omits must survive.
    updated = routes.update_case(suite["id"], created["id"], {"enabled": False, "tags": ["paused"]}, store=store)
    assert updated["enabled"] is False and updated["tags"] == ["paused"]
    assert updated["notes"] == "regression watch", "an omitted field is preserved, not cleared"
    assert updated["baseline_resolution"]["allowed_baseline_ids"] == [v1["id"], v2["id"]]

    # activateValidationBaseline / deactivateValidationBaseline return a baseline.
    activated = routes.activate(v1["id"], store=store)
    assert activated["id"] == v1["id"] and activated["status"] == "active"
    assert routes.deactivate(v1["id"], store=store)["status"] == "inactive"
    for field in ("version", "take_id", "pipeline_id", "source_run_id", "comparator", "comparator_source"):
        assert field in activated, f"ValidationBaseline declares {field}"

    # deleteValidationCase(suiteId, caseId) -> { deleted: string }
    assert routes.delete_case(suite["id"], created["id"], store=store) == {"deleted": created["id"]}
    assert store.get_suite(suite["id"])["cases"] == []


def test_case_routes_surface_refusals_as_http_errors(tmp_path: Path) -> None:
    """The UI needs a status code and a message, not a stack trace."""
    from fastapi import HTTPException

    from vision_3d_acquisition.api import validation as routes

    store, _, v1, _ = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = routes.add_case(suite["id"], {"baseline_ids": [v1["id"]]}, store=store)
    store.archive_suite(suite["id"])

    for call in (
        lambda: routes.add_case(suite["id"], {"baseline_ids": [v1["id"]]}, store=store),
        lambda: routes.update_case(suite["id"], case["id"], {"enabled": False}, store=store),
        lambda: routes.delete_case(suite["id"], case["id"], store=store),
    ):
        with pytest.raises(HTTPException) as raised:
            call()
        assert raised.value.status_code == 422
        assert "archived" in str(raised.value.detail)

    with pytest.raises(HTTPException) as raised:
        routes.activate("baseline_does_not_exist", store=store)
    assert raised.value.status_code == 404


def test_impact_route_reports_which_cases_would_follow_an_activation(tmp_path: Path) -> None:
    """The history drawer previews this before asking anyone to activate."""
    from fastapi import HTTPException

    from vision_3d_acquisition.api import validation as routes

    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    active = store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})
    pinned = store.add_case(
        suite["id"],
        {"baseline_ids": [v1["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": v1["id"]}},
    )

    impact = routes.resolution_impact(v2["id"], store=store)
    assert [x["case_id"] for x in impact["active_cases"]] == [active["id"]]
    assert [x["case_id"] for x in impact["pinned_cases_unchanged"]] == [pinned["id"]]

    with pytest.raises(HTTPException) as raised:
        routes.resolution_impact("baseline_does_not_exist", store=store)
    assert raised.value.status_code == 404
