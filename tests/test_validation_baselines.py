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


def test_coverage_reflects_the_resolved_baseline_not_every_configured_one(tmp_path: Path) -> None:
    """A case pinned to v1 must not also be counted against v2 and v3."""
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})

    pinned = store.add_case(
        suite["id"],
        {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "pinned", "baseline_id": v1["id"]}},
    )
    report = store.coverage(suite["id"])
    area = next(row for row in report["areas"] if row["area"] == v1["stage_id"])
    # v1 is inactive (v2 is active): a pin to it counts once, as stale — not twice.
    assert area["stale"] == 1
    assert area["approved"] == 0

    store.update_case(suite["id"], pinned["id"], {"baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], v2["id"]]}})
    report = store.coverage(suite["id"])
    area = next(row for row in report["areas"] if row["area"] == v1["stage_id"])
    # Allowed versions legitimately covers both configured versions.
    assert area["stale"] == 1 and area["approved"] == 1

    store.update_case(suite["id"], pinned["id"], {"baseline_resolution": {"mode": "active"}})
    report = store.coverage(suite["id"])
    area = next(row for row in report["areas"] if row["area"] == v1["stage_id"])
    # Active mode resolves to the one active version, not the configured (possibly stale) id.
    assert area["approved"] == 1 and area["stale"] == 0


def test_active_resolution_deduplicates_same_family_configured_twice(tmp_path: Path) -> None:
    """baseline_ids is frozen at add_case time and outlives mode switches, so it can
    legitimately hold two versions of one family. Both must resolve to one entry,
    not two — a duplicate would double-run the comparison in execute_suite()."""
    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(
        suite["id"],
        {"baseline_ids": [v1["id"], v2["id"]], "baseline_resolution": {"mode": "allowed_versions", "allowed_baseline_ids": [v1["id"], v2["id"]]}},
    )
    store.update_case(suite["id"], case["id"], {"baseline_resolution": {"mode": "active"}})

    resolved = store._case_baselines(_case_of(store, suite["id"], case["id"]))
    assert resolved == [v2["id"]], "both configured ids resolve to the same active sibling"

    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    assert len(execution["cases"][0]["comparisons"]) == 1, "the comparison must not run twice"


# --------------------------------------------------------------------------
# Matrix
# --------------------------------------------------------------------------

def _real_stage_fixture(tmp_path: Path) -> tuple[ValidationService, Path, dict]:
    """A baseline whose stage_id is one PIPELINE's real processing-unit registry
    actually reports, so it lands in a matrix column instead of vanishing --
    "segment" (the shared _resolved() fixture's stage) is not a registered
    stage for PIPELINE and so is invisible to matrix(), same as any other
    unrecognized stage id would be."""
    source = tmp_path / "candidate.png"
    Image.fromarray(_V1).save(source)
    store = ValidationService(_settings(tmp_path))
    resolved = {
        "artifact_root": tmp_path,
        "artifacts": [{"artifact_id": "final_object_mask", "stage_id": "detect_belt_plane", "processing_unit_id": "detect_belt_plane", "kind": "image", "path": source.name}],
        "result_payload": {"processing_pipeline": {"processing_units": []}, "recipe_snapshot": {}},
        "stage_params": {}, "recipe_id": None,
    }
    store._resolve = lambda **_: resolved  # type: ignore[method-assign]
    baseline = store.create_baselines({"take_id": "take_1", "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "detect_belt_plane", "artifact_id": "final_object_mask"})["baselines"][0]
    Image.fromarray(_V2).save(source)
    return store, source, baseline



def test_matrix_groups_by_stage_and_reports_missing_baseline_elsewhere(tmp_path: Path) -> None:
    store, source, baseline = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    grid = store.matrix(execution["id"])
    assert grid["execution_id"] == execution["id"]
    row = grid["rows"][0]
    assert row["case_id"] == case["id"] and row["take_id"] == "take_1"

    by_stage = {column["stage_id"]: cell for column, cell in zip(grid["columns"], row["cells"])}
    covered = by_stage[baseline["stage_id"]]
    assert covered["status"] == "regression"
    assert covered["comparison_ids"] == [baseline["id"]]

    uncovered = [cell for stage, cell in by_stage.items() if stage != baseline["stage_id"]]
    assert uncovered and all(cell["status"] == "missing_baseline" for cell in uncovered)
    assert all(cell["comparison_ids"] == [] for cell in uncovered)


def test_matrix_marks_first_divergence_on_its_own_column_only(tmp_path: Path) -> None:
    store, source, baseline = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    grid = store.matrix(execution["id"])
    row = grid["rows"][0]
    flagged = [cell for cell in row["cells"] if cell["first_divergence"]]
    assert len(flagged) == 1
    assert flagged[0]["artifact_id"] == baseline["artifact_id"]
    others = [cell for cell in row["cells"] if not cell["first_divergence"]]
    assert all(cell["artifact_id"] is None for cell in others)


def test_matrix_falls_back_to_a_fixed_stage_list_for_an_unregistered_pipeline(tmp_path: Path) -> None:
    """No processing-unit definitions exist for this pipeline id, so the column
    set must still be usable rather than empty."""
    store = ValidationService(_settings(tmp_path))
    suite = store.create_suite({"name": "s", "pipeline_id": "no_such_pipeline"})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    grid = store.matrix(execution["id"])
    assert grid["rows"] == []
    assert [c["stage_id"] for c in grid["columns"]] == [
        "detect_belt_plane", "normalize_heights_to_plane", "remove_belt_segment_objects",
        "extract_connected_components", "fit_object_geometry", "compute_height_metrics",
        "classification", "overlay",
    ]


def test_matrix_reads_each_baseline_at_most_once(tmp_path: Path) -> None:
    """Regression guard for the N+1 fix: matrix() used to call get_baseline()
    once per (case, column, comparison), reading the same handful of baseline
    records off disk hundreds of times over for a suite with several cases."""
    store, source, baseline = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    for _ in range(5):
        store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    calls = {"n": 0}
    real_get_baseline = store.get_baseline

    def _counting_get_baseline(baseline_id):
        calls["n"] += 1
        return real_get_baseline(baseline_id)

    store.get_baseline = _counting_get_baseline  # type: ignore[method-assign]
    grid = store.matrix(execution["id"])
    store.get_baseline = real_get_baseline  # type: ignore[method-assign]

    # One comparison per case, all against the same baseline (v2): five cases,
    # nine columns each in the fallback list -- the naive loop would have
    # called get_baseline() up to 45 times for a single baseline record.
    assert calls["n"] <= 5, f"expected at most one read per case, got {calls['n']}"
    assert len(grid["rows"]) == 5


# --------------------------------------------------------------------------
# Integrity verification: every unhealthy branch, not just the clean path
# --------------------------------------------------------------------------

def test_verify_integrity_detects_a_missing_snapshot(tmp_path: Path) -> None:
    store, source, v1, v2 = _resolution_fixture(tmp_path)
    snapshot = store.root / "baselines" / v1["id"] / v1["snapshot_artifact_path"]
    snapshot.unlink()

    report = store.verify_integrity()
    assert report["ok"] is False
    assert {"kind": "missing_snapshot", "baseline_id": v1["id"]} in report["problems"]


def test_verify_integrity_detects_a_checksum_mismatch(tmp_path: Path) -> None:
    store, source, v1, v2 = _resolution_fixture(tmp_path)
    snapshot = store.root / "baselines" / v1["id"] / v1["snapshot_artifact_path"]
    snapshot.write_bytes(b"corrupted, does not match the recorded checksum")

    report = store.verify_integrity()
    assert report["ok"] is False
    assert {"kind": "checksum_mismatch", "baseline_id": v1["id"]} in report["problems"]


def test_verify_integrity_detects_a_snapshot_path_that_escapes_the_baselines_root(tmp_path: Path) -> None:
    store, source, v1, v2 = _resolution_fixture(tmp_path)
    record_path = store.root / "baselines" / v1["id"] / "baseline.json"
    record = json.loads(record_path.read_text())
    record["snapshot_artifact_path"] = "../../../../../../etc/passwd"
    record_path.write_text(json.dumps(record))

    report = store.verify_integrity()
    assert report["ok"] is False
    assert {"kind": "snapshot_path_escape", "baseline_id": v1["id"]} in report["problems"]


def test_verify_integrity_detects_an_unparseable_baseline_record(tmp_path: Path) -> None:
    store, source, v1, v2 = _resolution_fixture(tmp_path)
    stray = store.root / "baselines" / "baseline_not_a_dict"
    stray.mkdir()
    (stray / "baseline.json").write_text(json.dumps(["not", "a", "dict"]))

    report = store.verify_integrity()
    assert report["ok"] is False
    assert any(p["kind"] == "invalid_baseline_record" for p in report["problems"])
    # Two genuine baselines exist; the one unparseable record is not counted as a third.
    assert report["baseline_count"] == 2


def test_verify_integrity_detects_a_suite_case_referencing_a_baseline_that_no_longer_exists(tmp_path: Path) -> None:
    store, source, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})

    # Simulate the baseline vanishing without going through delete_case: hand
    # corruption, not a reachable API path, which is exactly what this check guards.
    import shutil
    shutil.rmtree(store.root / "baselines" / v1["id"])

    report = store.verify_integrity()
    assert report["ok"] is False
    assert {"kind": "broken_suite_baseline", "case_id": case["id"], "baseline_id": v1["id"]} in report["problems"]


# --------------------------------------------------------------------------
# Conservative object matching: the centroid fallback and the ambiguity guard
# --------------------------------------------------------------------------

def test_centroid_matching_requires_a_clear_winner_within_5mm(tmp_path: Path) -> None:
    store = ValidationService(_settings(tmp_path))
    left = [{"id": "a", "centroid": [0.0, 0.0, 0.0]}]

    # Well within 5mm and unambiguous: matches.
    close = store._match_objects(left, [{"id": "b", "centroid": [3.0, 0.0, 0.0]}])
    assert len(close["matches"]) == 1
    assert close["matches"][0]["match_method"] == "centroid"
    assert close["matches"][0]["candidate_object_id"] == "b"

    # Outside 5mm: no match at all, not a forced best-effort pairing.
    far = store._match_objects(left, [{"id": "b", "centroid": [8.0, 0.0, 0.0]}])
    assert far["matches"] == []
    assert far["baseline_only"] == [{"index": 0, "object_id": "a"}]


def test_centroid_matching_refuses_to_guess_between_two_close_candidates(tmp_path: Path) -> None:
    """Two candidates within 1mm of each other near the baseline object: too close
    to call, so it is reported as ambiguous rather than picking the nearer one."""
    store = ValidationService(_settings(tmp_path))
    left = [{"id": "a", "centroid": [0.0, 0.0, 0.0]}]
    right = [
        {"id": "b", "centroid": [3.0, 0.0, 0.0]},
        {"id": "c", "centroid": [3.5, 0.0, 0.0]},
    ]

    result = store._match_objects(left, right)
    assert result["matches"] == []
    assert len(result["ambiguous"]) == 1
    assert result["ambiguous"][0]["baseline_object_id"] == "a"
    assert set(result["ambiguous"][0]["candidate_indices"]) == {0, 1}


def test_stable_id_match_outranks_centroid_even_when_geometry_disagrees(tmp_path: Path) -> None:
    store = ValidationService(_settings(tmp_path))
    left = [{"id": "a", "centroid": [0.0, 0.0, 0.0]}]
    right = [{"id": "a", "centroid": [50.0, 50.0, 50.0]}]

    result = store._match_objects(left, right)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["match_method"] == "stable_id"
    assert result["matches"][0]["match_score"] == 1.0


# --------------------------------------------------------------------------
# Route-layer logic beyond a 1:1 passthrough to the service
# --------------------------------------------------------------------------

def test_get_baseline_route_404s_on_a_missing_id_without_raising_something_else(tmp_path: Path) -> None:
    from fastapi import HTTPException
    from vision_3d_acquisition.api import validation as routes

    store, _, v1, v2 = _resolution_fixture(tmp_path)
    assert routes.get_baseline(v1["id"], store=store)["id"] == v1["id"]

    with pytest.raises(HTTPException) as raised:
        routes.get_baseline("baseline_does_not_exist", store=store)
    assert raised.value.status_code == 404


def test_suite_executions_route_filters_to_the_requested_suite_only(tmp_path: Path) -> None:
    from vision_3d_acquisition.api import validation as routes

    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite_a = store.create_suite({"name": "a", "pipeline_id": PIPELINE})
    suite_b = store.create_suite({"name": "b", "pipeline_id": PIPELINE})
    store.add_case(suite_a["id"], {"baseline_ids": [v1["id"]]})
    store.add_case(suite_b["id"], {"baseline_ids": [v1["id"]]})
    store.execute_suite(suite_a["id"], {"candidate_run_id": "run_1"})
    store.execute_suite(suite_b["id"], {"candidate_run_id": "run_1"})

    executions = routes.suite_executions(suite_a["id"], store=store)
    assert len(executions) == 1
    assert all(e["suite_id"] == suite_a["id"] for e in executions)


def test_progress_route_returns_the_progress_of_a_completed_execution(tmp_path: Path) -> None:
    from vision_3d_acquisition.api import validation as routes

    store, _, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    progress = routes.progress(execution["id"], store=store)
    assert progress["completed"] == progress["total"] == 1
    assert progress["status"] in {"completed", "completed_with_regressions"}


def test_promote_route_extracts_case_id_from_the_payload(tmp_path: Path) -> None:
    from fastapi import HTTPException
    from vision_3d_acquisition.api import validation as routes

    store, _, v1, case, execution = _promotion_fixture(tmp_path)
    outcome = routes.promote(execution["id"], v1["id"], {"case_id": case["id"], "activate": True}, store=store)
    assert outcome["baseline"]["version"] == 2

    with pytest.raises(HTTPException) as raised:
        routes.promote(execution["id"], v1["id"], {"activate": True}, store=store)
    assert raised.value.status_code == 404, "an absent case_id must not silently match some other case"
def test_unmatched_objects_carry_their_identity_not_just_a_position(tmp_path: Path) -> None:
    """A detail view has to name a missing or added object, not just count it."""
    store = ValidationService(_settings(tmp_path))
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps({"objects": [
        {"object_id": "one", "diameter_mm": 10.0},
        {"object_id": "gone_missing", "diameter_mm": 12.0},
    ]}))
    right.write_text(json.dumps({"objects": [
        {"object_id": "one", "diameter_mm": 10.1},
        {"object_id": "newly_added", "diameter_mm": 8.0},
    ]}))
    base = {"id": "m", "version": 1, "comparison_policy": {"metrics": {"diameter_mm": {"absolute_tolerance": 5}}}}
    result = store._measurement_compare(left, right, base, {"artifact_id": "measure"})

    assert result["matching"]["baseline_only"] == [{"index": 1, "object_id": "gone_missing"}]
    assert result["matching"]["candidate_only"] == [{"index": 1, "object_id": "newly_added"}]
    assert result["metrics"]["missing_object_count"] == 1
    assert result["metrics"]["added_object_count"] == 1


def test_file_route_serves_diff_artifacts_and_rejects_traversal(tmp_path: Path) -> None:
    from fastapi import HTTPException
    from vision_3d_acquisition.api import validation as routes

    store, source, v1, v2 = _resolution_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    case = store.add_case(suite["id"], {"baseline_ids": [v1["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    diff_path = execution["cases"][0]["comparisons"][0]["diff_artifacts"][0]

    served = routes.get_file(diff_path, store=store)
    assert Path(served.path) == (store.root / diff_path).resolve()

    for attempt in ("../../../etc/passwd", "..%2f..%2fetc/passwd", "", ".", "baselines/../../../../etc/passwd"):
        with pytest.raises(HTTPException) as raised:
            routes.get_file(attempt, store=store)
        assert raised.value.status_code == 404

    with pytest.raises(HTTPException):
        routes.get_file("baselines/does_not_exist.png", store=store)


# --------------------------------------------------------------------------
# Stage summary: quantitative rollups across a sample, per stage and comparator
# --------------------------------------------------------------------------

def test_stage_summary_aggregates_a_known_metric_correctly(tmp_path: Path) -> None:
    """Three cases at one stage, one known IoU each: mean/median/min/max/p95 must
    match what those three numbers actually produce, not an approximation."""
    store, source, base1 = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [base1["id"]]})

    # Two more cases, same stage and comparator, different takes.
    for take_id, source_path in (("take_2", tmp_path / "c2.png"), ("take_3", tmp_path / "c3.png")):
        Image.fromarray(_V1).save(source_path)
        resolved = {
            "artifact_root": tmp_path,
            "artifacts": [{"artifact_id": "final_object_mask", "stage_id": "detect_belt_plane", "processing_unit_id": "detect_belt_plane", "kind": "image", "path": source_path.name}],
            "result_payload": {"processing_pipeline": {"processing_units": []}, "recipe_snapshot": {}},
            "stage_params": {}, "recipe_id": None,
        }
        store._resolve = lambda **_: resolved  # type: ignore[method-assign]
        baseline = store.create_baselines({"take_id": take_id, "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "detect_belt_plane", "artifact_id": "final_object_mask"})["baselines"][0]
        store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})

    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    exec_path = store.root / "executions" / execution["id"] / "execution.json"

    # Overwrite with three exact, hand-picked IoU values so the statistics are
    # verifiable by hand rather than by re-deriving them from pixels.
    record = json.loads(exec_path.read_text())
    known = [1.0, 0.9, 0.5]
    for case, iou in zip(record["cases"], known):
        case["comparisons"][0]["metrics"]["iou"] = iou
    exec_path.write_text(json.dumps(record))

    summary = store.stage_summary(execution["id"])
    stage = next(s for s in summary["stages"] if s["stage_id"] == "detect_belt_plane")
    bucket = next(c for c in stage["comparators"] if c["comparator"] == "binary_mask")
    iou_stats = bucket["metrics"]["iou"]

    assert bucket["count"] == 3
    assert iou_stats["count"] == 3
    assert iou_stats["mean"] == pytest.approx(sum(known) / 3)
    assert iou_stats["median"] == pytest.approx(0.9)
    assert iou_stats["min"] == pytest.approx(0.5)
    assert iou_stats["max"] == pytest.approx(1.0)
    assert iou_stats["p95"] == pytest.approx(float(__import__("numpy").percentile(known, 95)))


def test_stage_summary_never_pools_different_comparators_at_the_same_stage(tmp_path: Path) -> None:
    """A mask's IoU and a raster's MAE must never be averaged together."""
    store, source, mask_baseline = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [mask_baseline["id"]]})

    raster_source = tmp_path / "raster.npy"
    import numpy as np
    np.save(raster_source, np.array([[1.0, 2.0]], dtype=np.float32))
    resolved = {
        "artifact_root": tmp_path,
        "artifacts": [{"artifact_id": "final_object_mask", "stage_id": "detect_belt_plane", "processing_unit_id": "detect_belt_plane", "kind": "image", "path": raster_source.name}],
        "result_payload": {"processing_pipeline": {"processing_units": []}, "recipe_snapshot": {}},
        "stage_params": {}, "recipe_id": None,
    }
    store._resolve = lambda **_: resolved  # type: ignore[method-assign]
    raster_baseline = store.create_baselines({"take_id": "take_raster", "pipeline_id": PIPELINE, "run_id": "run_1", "approval_scope": "artifact", "stage_id": "detect_belt_plane", "artifact_id": "final_object_mask", "comparison_policy": {"comparator": "numeric_raster"}})["baselines"][0]
    store.add_case(suite["id"], {"baseline_ids": [raster_baseline["id"]]})

    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    summary = store.stage_summary(execution["id"])
    stage = next(s for s in summary["stages"] if s["stage_id"] == "detect_belt_plane")
    comparators = {c["comparator"] for c in stage["comparators"]}

    assert comparators == {"binary_mask", "numeric_raster"}
    mask_bucket = next(c for c in stage["comparators"] if c["comparator"] == "binary_mask")
    raster_bucket = next(c for c in stage["comparators"] if c["comparator"] == "numeric_raster")
    assert mask_bucket["count"] == 1 and raster_bucket["count"] == 1
    assert "mae" not in mask_bucket["metrics"]
    assert "iou" not in raster_bucket["metrics"]


def test_stage_summary_excludes_disabled_cases_and_shows_stages_with_no_data(tmp_path: Path) -> None:
    store, source, baseline = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [baseline["id"]], "enabled": False})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    summary = store.stage_summary(execution["id"])
    stage = next(s for s in summary["stages"] if s["stage_id"] == "detect_belt_plane")
    assert stage["comparators"] == [], "the only case is disabled, so this stage has no data"

    # Every registered stage still appears, not just the ones with comparisons.
    stage_ids = {s["stage_id"] for s in summary["stages"]}
    assert len(stage_ids) > 1
    assert all(s["comparators"] == [] for s in summary["stages"] if s["stage_id"] != "detect_belt_plane")


def test_stage_summary_counts_regressions_alongside_the_numbers(tmp_path: Path) -> None:
    store, source, baseline = _real_stage_fixture(tmp_path)  # candidate already diverges from the baseline
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})
    assert execution["cases"][0]["status"] == "regression"

    summary = store.stage_summary(execution["id"])
    stage = next(s for s in summary["stages"] if s["stage_id"] == baseline["stage_id"])
    bucket = stage["comparators"][0]
    assert bucket["status_counts"] == {"regression": 1}


def test_stage_summary_treats_a_boolean_metric_as_a_pass_fraction(tmp_path: Path) -> None:
    """json_fields reports {"equal": bool}; averaging that across cases is exactly
    the fraction that passed, which is the whole point of a boolean metric."""
    store = ValidationService(_settings(tmp_path))
    left, right = tmp_path / "l.json", tmp_path / "r.json"
    left.write_text(json.dumps({"a": 1}))
    right.write_text(json.dumps({"a": 1}))
    base = {"id": "b", "version": 1, "comparison_policy": {}}
    equal = store._json_compare(left, right, base, {"artifact_id": "x"}, "json_fields")
    assert equal["metrics"] == {"equal": True}
    unequal = {**equal, "metrics": {"equal": False}}

    for comparison in (equal, unequal):
        comparison["baseline_ref"] = {"baseline_id": "b1"}
    store.get_baseline = lambda _id: {"stage_id": "classification"}  # type: ignore[method-assign]

    execution_id = "execution_synthetic"
    execution = {
        "id": execution_id,
        "cases": [
            {"case_id": "c1", "comparisons": [equal]},
            {"case_id": "c2", "comparisons": [unequal]},
            {"case_id": "c3", "comparisons": [equal]},
        ],
    }
    exec_path = store.root / "executions" / execution_id / "execution.json"
    exec_path.parent.mkdir(parents=True)
    exec_path.write_text(json.dumps(execution))

    summary = store.stage_summary(execution_id)
    stage = next(s for s in summary["stages"] if s["stage_id"] == "classification")
    bucket = stage["comparators"][0]
    # Two of three passed: the mean of {1, 0, 1} is the pass fraction, 2/3.
    assert bucket["metrics"]["equal"]["mean"] == pytest.approx(2 / 3)
    assert bucket["metrics"]["equal"]["count"] == 3


def test_stage_summary_route_matches_the_service_method(tmp_path: Path) -> None:
    from vision_3d_acquisition.api import validation as routes

    store, source, baseline = _real_stage_fixture(tmp_path)
    suite = store.create_suite({"name": "s", "pipeline_id": PIPELINE})
    store.add_case(suite["id"], {"baseline_ids": [baseline["id"]]})
    execution = store.execute_suite(suite["id"], {"candidate_run_id": "run_1"})

    assert routes.stage_summary(execution["id"], store=store) == store.stage_summary(execution["id"])
