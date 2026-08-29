from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

from vision_3d_acquisition.pipelines.processing_units import processing_unit_contract_fingerprint


PIPELINE_STAGE_ORDER = [
    "input",
    "detect_belt_plane",
    "normalize_heights_to_plane",
    "remove_belt_segment_objects",
    "geometry",
    "measurement",
    "measurement_diagnostics",
    "classification",
    "overlay",
]


@dataclass(frozen=True)
class ProcessingUnitRerunCapability:
    unit_id: str
    stage_id: str
    unit_kind: str
    classification: str
    boundary_stage_only: bool
    independently_rerunnable: bool
    optional_source_dependent: bool = False


@dataclass
class PartialRerunPlan:
    safe: bool
    mode: str
    selected_unit_id: str
    selected_stage_id: str | None
    effective_start_stage_id: str | None
    execution_boundary_unit_id: str | None = None
    selection_policy: str = "stage_boundary_only_v1"
    suggested_mode: str = "full_run"
    fallback_mode: str | None = None
    stages_to_reuse: list[str] = field(default_factory=list)
    stages_to_rerun: list[str] = field(default_factory=list)
    stages_invalidated: list[str] = field(default_factory=list)
    units_to_reuse: list[str] = field(default_factory=list)
    units_to_rerun: list[str] = field(default_factory=list)
    units_invalidated: list[str] = field(default_factory=list)
    required_missing_artifacts: list[str] = field(default_factory=list)
    reusable_artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    changed_unit_ids: list[str] = field(default_factory=list)
    changed_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    contract_fingerprint_match: bool | None = None
    calibration_match: bool | None = None
    parent_trace_source: str | None = None
    explanation: dict[str, Any] = field(default_factory=dict)
    plan_quality: str = "unsafe"
    confidence: str = "low"
    plan_id: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _base_unsafe_plan(pipeline_id: str, selected_unit_id: str, *, reason: str) -> PartialRerunPlan:
    plan = PartialRerunPlan(
        safe=False,
        mode="full_run",
        selected_unit_id=str(selected_unit_id),
        selected_stage_id=None,
        effective_start_stage_id=None,
        fallback_mode="full_run",
    )
    plan.blocking_reasons.append(reason)
    plan.explanation = {
        "summary": reason,
        "selected_unit_reason": f"Planner could not validate selected unit {selected_unit_id}.",
        "boundary_reason": None,
        "reuse_reason": None,
        "rerun_reason": None,
        "fallback_reason": "Planner selected full_run because the request could not be validated safely.",
    }
    plan.plan_quality = "unsafe"
    plan.confidence = "low"
    return plan


def _unit_index(units: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(unit.get("id") or ""): unit
        for unit in units
        if isinstance(unit, Mapping) and unit.get("id")
    }


def _stage_roots(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots = [
        unit
        for unit in units
        if isinstance(unit, Mapping) and str(unit.get("kind") or "") == "stage"
    ]
    return sorted(roots, key=lambda item: (PIPELINE_STAGE_ORDER.index(str(item.get("stage_id"))) if str(item.get("stage_id")) in PIPELINE_STAGE_ORDER else 999, int(item.get("order") or 0)))


def _stage_units(units: list[dict[str, Any]], stage_id: str) -> list[dict[str, Any]]:
    return [
        unit
        for unit in units
        if isinstance(unit, Mapping) and str(unit.get("stage_id") or "") == stage_id
    ]


def _classify_unit(unit: Mapping[str, Any]) -> ProcessingUnitRerunCapability:
    unit_id = str(unit.get("id") or "")
    stage_id = str(unit.get("stage_id") or "")
    unit_kind = str(unit.get("kind") or "")
    outputs = [item for item in unit.get("outputs") or [] if isinstance(item, Mapping) and item.get("artifact_id")]
    parameters = [item for item in unit.get("parameters") or [] if isinstance(item, Mapping)]
    diagnostics = [item for item in unit.get("diagnostics") or [] if item]
    optional_source_dependent = unit_id == "input.reflectance_rgb"
    if unit_kind == "stage":
        classification = "logical_container"
    elif optional_source_dependent:
        classification = "optional_source_dependent"
    elif outputs:
        classification = "artifact_emitting"
    elif diagnostics:
        classification = "diagnostic_only"
    elif parameters:
        classification = "parameter_only"
    else:
        classification = "executable_unit"
    return ProcessingUnitRerunCapability(
        unit_id=unit_id,
        stage_id=stage_id,
        unit_kind=unit_kind,
        classification=classification,
        boundary_stage_only=unit_kind != "stage",
        independently_rerunnable=bool(unit.get("supports_partial_rerun") is True),
        optional_source_dependent=optional_source_dependent,
    )


def _artifact_ids(result: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(result, Mapping):
        return set()
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        return set()
    return {
        str(artifact.get("artifact_id"))
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("artifact_id")
    }


def _trace_units(result: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(result, Mapping):
        return {}
    trace = result.get("processing_unit_trace")
    if not isinstance(trace, Mapping):
        return {}
    raw = trace.get("units")
    if not isinstance(raw, Mapping):
        raw = trace.get("unit_results")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(unit_id): dict(payload)
        for unit_id, payload in raw.items()
        if isinstance(payload, Mapping)
    }


def _recipe_snapshot(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    snapshot = result.get("recipe_snapshot")
    return dict(snapshot) if isinstance(snapshot, Mapping) else {}


def _parameters_by_unit_from_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(snapshot, Mapping):
        return {}
    raw = snapshot.get("parameters_by_unit")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(unit_id): dict(values)
        for unit_id, values in raw.items()
        if isinstance(values, Mapping)
    }


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _resolve_changed_parameters(
    changed_parameters: Mapping[str, Any] | None,
    *,
    current_recipe_snapshot: Mapping[str, Any] | None,
    parent_run_result: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if isinstance(changed_parameters, Mapping):
        candidate = {
            str(unit_id): dict(values)
            for unit_id, values in changed_parameters.items()
            if isinstance(values, Mapping)
        }
    else:
        candidate = _parameters_by_unit_from_snapshot(current_recipe_snapshot)
    if not candidate:
        return {}
    parent_snapshot = _recipe_snapshot(parent_run_result)
    parent_values = _parameters_by_unit_from_snapshot(parent_snapshot)
    resolved: dict[str, dict[str, Any]] = {}
    for unit_id, values in candidate.items():
        parent_unit_values = parent_values.get(unit_id, {})
        changed_values = {
            key: value
            for key, value in values.items()
            if _stable_value(value) != _stable_value(parent_unit_values.get(key))
        }
        if changed_values:
            resolved[unit_id] = changed_values
    return resolved


def _stage_order_index(stage_id: str) -> int:
    try:
        return PIPELINE_STAGE_ORDER.index(stage_id)
    except ValueError:
        return 999


def _required_input_artifacts_for_stage(units: list[dict[str, Any]], stage_id: str) -> list[str]:
    root = next(
        (
            unit
            for unit in units
            if isinstance(unit, Mapping)
            and str(unit.get("id") or "") == stage_id
            and str(unit.get("kind") or "") == "stage"
        ),
        None,
    )
    if not isinstance(root, Mapping):
        return []
    required: list[str] = []
    for item in root.get("inputs") or []:
        if not isinstance(item, Mapping):
            continue
        artifact_id = item.get("artifact_id")
        if artifact_id and item.get("required", True) is not False:
            required.append(str(artifact_id))
    return required


def _calibration_reference(payload: Mapping[str, Any] | None) -> Any:
    if not isinstance(payload, Mapping):
        return None
    calibration = payload.get("calibration_snapshot_reference")
    if isinstance(calibration, Mapping) and calibration:
        return dict(calibration)
    profile_id = payload.get("calibration_profile_id")
    return {"calibration_profile_id": profile_id} if profile_id else None


def _build_explanation(plan: PartialRerunPlan) -> dict[str, Any]:
    selected_unit_reason = (
        f"Selected unit {plan.selected_unit_id} is a substage and partial rerun v1 plans at public stage boundaries."
        if plan.execution_boundary_unit_id and plan.execution_boundary_unit_id != plan.selected_unit_id
        else f"Selected unit {plan.selected_unit_id} is planned directly."
    )
    boundary_reason = (
        f"Selected substage is elevated to public stage {plan.execution_boundary_unit_id} for v1 safety."
        if plan.execution_boundary_unit_id
        else None
    )
    reuse_reason = (
        f"Upstream stages can be reused: {', '.join(plan.stages_to_reuse)}."
        if plan.stages_to_reuse
        else "No upstream stages can be safely reused."
    )
    rerun_reason = (
        f"The selected boundary and downstream stages must rerun: {', '.join(plan.stages_to_rerun)}."
        if plan.stages_to_rerun
        else "No rerun stages were selected."
    )
    fallback_reason = (
        "Full-run fallback was selected because reuse safety checks failed."
        if not plan.safe
        else None
    )
    if plan.safe:
        summary = (
            f"Changing {plan.selected_unit_id} requires rerunning {', '.join(plan.stages_to_rerun)}. "
            f"Reusable upstream stages: {', '.join(plan.stages_to_reuse) or 'none'}."
        )
    else:
        summary = (
            f"Planner could not safely reuse upstream results for {plan.selected_unit_id}. "
            f"Fallback mode is {plan.fallback_mode or 'full_run'}."
        )
    return {
        "summary": summary,
        "selected_unit_reason": selected_unit_reason,
        "boundary_reason": boundary_reason,
        "reuse_reason": reuse_reason,
        "rerun_reason": rerun_reason,
        "fallback_reason": fallback_reason,
    }


def _plan_quality_and_confidence(plan: PartialRerunPlan) -> tuple[str, str]:
    if not plan.safe:
        return "unsafe", "low"
    if plan.required_missing_artifacts:
        return "unsafe", "low"
    if plan.warnings:
        return "conservative", "medium"
    if plan.parent_trace_source and plan.parent_trace_source != "runtime_unit_callbacks":
        return "conservative", "medium"
    return "exact", "high"


def finalize_partial_rerun_plan(plan: PartialRerunPlan) -> dict[str, Any]:
    plan.safe = not plan.blocking_reasons
    plan.mode = "rerun_from_unit" if plan.safe else "full_run"
    plan.suggested_mode = "rerun_from_unit" if plan.safe else "full_run"
    plan.fallback_mode = None if plan.safe else "full_run"
    plan.explanation = _build_explanation(plan)
    plan.plan_quality, plan.confidence = _plan_quality_and_confidence(plan)
    return plan.to_dict()


def persist_partial_rerun_plan(data_dir: Path, pipeline_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    plan_id = str(plan.get("plan_id") or f"prp_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}")
    created_at = str(plan.get("created_at") or _now_iso())
    root = data_dir / "partial_rerun_plans" / plan_id
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        **dict(plan),
        "plan_id": plan_id,
        "created_at": created_at,
        "pipeline_id": pipeline_id,
    }
    path = root / "plan.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        **payload,
        "plan_path": str(path.relative_to(data_dir)),
    }


def list_partial_rerun_plans(data_dir: Path, pipeline_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
    root = data_dir / "partial_rerun_plans"
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in root.glob("*/plan.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("pipeline_id") or "") != pipeline_id:
            continue
        entries.append({
            "plan_id": payload.get("plan_id"),
            "created_at": payload.get("created_at"),
            "pipeline_id": payload.get("pipeline_id"),
            "selected_unit_id": payload.get("selected_unit_id"),
            "execution_boundary_unit_id": payload.get("execution_boundary_unit_id"),
            "safe": payload.get("safe"),
            "mode": payload.get("mode"),
            "fallback_mode": payload.get("fallback_mode"),
            "plan_quality": payload.get("plan_quality"),
            "confidence": payload.get("confidence"),
            "plan_path": str(path.relative_to(data_dir)),
        })
    entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return entries[: max(1, limit)]


def plan_partial_rerun(
    pipeline_id: str,
    selected_unit_id: str,
    changed_parameters: Mapping[str, Any] | None,
    current_recipe_snapshot: Mapping[str, Any] | None,
    parent_run_result: Mapping[str, Any] | None,
    processing_unit_contracts: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved_changed_parameters = _resolve_changed_parameters(
        changed_parameters,
        current_recipe_snapshot=current_recipe_snapshot,
        parent_run_result=parent_run_result,
    )
    plan = PartialRerunPlan(
        safe=False,
        mode="full_run",
        selected_unit_id=str(selected_unit_id),
        selected_stage_id=None,
        effective_start_stage_id=None,
        changed_unit_ids=sorted(resolved_changed_parameters.keys()),
        changed_parameters=resolved_changed_parameters,
        fallback_mode="full_run",
    )
    if pipeline_id != "mining_steel_ball_classification_25d":
        plan.blocking_reasons.append(f"Unsupported pipeline for partial rerun planning: {pipeline_id}")
        return finalize_partial_rerun_plan(plan)

    units = [dict(unit) for unit in processing_unit_contracts if isinstance(unit, Mapping)]
    unit_by_id = _unit_index(units)
    selected_unit = unit_by_id.get(str(selected_unit_id))
    if selected_unit is None:
        plan.blocking_reasons.append(f"Unknown processing unit: {selected_unit_id}")
        return finalize_partial_rerun_plan(plan)

    selected_stage_id = str(selected_unit.get("stage_id") or "") or None
    plan.selected_stage_id = selected_stage_id
    plan.parent_trace_source = str((parent_run_result or {}).get("processing_unit_trace", {}).get("trace_source") or "") if isinstance(parent_run_result, Mapping) else None

    capability = _classify_unit(selected_unit)
    if capability.unit_kind == "stage":
        plan.blocking_reasons.append("Selected unit is a logical container; choose a concrete substage for planning.")
        return finalize_partial_rerun_plan(plan)

    plan.effective_start_stage_id = selected_stage_id
    plan.execution_boundary_unit_id = selected_stage_id
    if selected_stage_id:
        plan.warnings.append(
            f"Selected substage {selected_unit_id} is planned at public stage boundary {selected_stage_id}."
        )

    current_fingerprint = processing_unit_contract_fingerprint(units)
    parent_snapshot = _recipe_snapshot(parent_run_result)
    current_snapshot = dict(current_recipe_snapshot) if isinstance(current_recipe_snapshot, Mapping) else {}
    parent_fingerprint = str(parent_snapshot.get("processing_unit_contract_fingerprint") or "")
    current_snapshot_fingerprint = str(current_snapshot.get("processing_unit_contract_fingerprint") or "")
    plan.contract_fingerprint_match = None
    if parent_fingerprint:
        plan.contract_fingerprint_match = parent_fingerprint == current_fingerprint
        if not plan.contract_fingerprint_match:
            plan.blocking_reasons.append("Parent run contract fingerprint does not match current processing-unit contract.")
    if current_snapshot_fingerprint and current_snapshot_fingerprint != current_fingerprint:
        plan.blocking_reasons.append("Current recipe snapshot fingerprint does not match current processing-unit contract.")

    parent_calibration = _calibration_reference(parent_snapshot) or _calibration_reference(parent_run_result)
    current_calibration = _calibration_reference(current_snapshot)
    if parent_calibration is not None and current_calibration is not None:
        plan.calibration_match = parent_calibration == current_calibration
        if not plan.calibration_match:
            plan.blocking_reasons.append("Calibration snapshot/reference mismatch blocks upstream artifact reuse.")

    stage_roots = _stage_roots(units)
    ordered_stage_ids = [str(root.get("stage_id") or "") for root in stage_roots if root.get("stage_id")]
    if selected_stage_id not in ordered_stage_ids:
        plan.blocking_reasons.append(f"Selected stage {selected_stage_id} is not a known public stage boundary.")
        return finalize_partial_rerun_plan(plan)
    selected_stage_index = ordered_stage_ids.index(selected_stage_id)

    changed_units = plan.changed_unit_ids
    upstream_changed_units = [
        unit_id
        for unit_id in changed_units
        if _stage_order_index(str(unit_by_id.get(unit_id, {}).get("stage_id") or "")) < selected_stage_index
    ]
    if upstream_changed_units:
        plan.blocking_reasons.append(
            "Changed parameters affect upstream stages; conservative v1 planner falls back to full rerun."
        )

    plan.stages_to_reuse = ordered_stage_ids[:selected_stage_index]
    plan.stages_to_rerun = ordered_stage_ids[selected_stage_index:]
    plan.stages_invalidated = list(plan.stages_to_rerun)
    plan.units_to_reuse = [
        str(unit.get("id") or "")
        for stage_id in plan.stages_to_reuse
        for unit in _stage_units(units, stage_id)
        if unit.get("id")
    ]
    plan.units_to_rerun = [
        str(unit.get("id") or "")
        for stage_id in plan.stages_to_rerun
        for unit in _stage_units(units, stage_id)
        if unit.get("id")
    ]
    plan.units_invalidated = list(plan.units_to_rerun)

    available_artifacts = _artifact_ids(parent_run_result)
    plan.reusable_artifacts = sorted(available_artifacts)
    required_inputs = _required_input_artifacts_for_stage(units, selected_stage_id)
    plan.required_missing_artifacts = sorted(
        artifact_id for artifact_id in required_inputs if artifact_id not in available_artifacts
    )
    if plan.required_missing_artifacts:
        plan.blocking_reasons.append(
            f"Missing required upstream artifacts for stage {selected_stage_id}: {', '.join(plan.required_missing_artifacts)}"
        )

    trace_units = _trace_units(parent_run_result)
    unsafe_reused_units = [
        unit_id
        for unit_id in plan.units_to_reuse
        if str(trace_units.get(unit_id, {}).get("status") or "") in {"failed", "invalidated", "pending"}
    ]
    if unsafe_reused_units:
        plan.blocking_reasons.append(
            f"Upstream units are not reusable because parent trace is not trustworthy: {', '.join(unsafe_reused_units)}"
        )
    if not trace_units:
        plan.warnings.append("Parent run has no runtime trace payload; planner is relying on artifact presence only.")
    elif str((parent_run_result or {}).get("processing_unit_trace", {}).get("trace_source") or "") != "runtime_unit_callbacks":
        plan.warnings.append("Parent run trace is inferred or mixed; reuse confidence is reduced.")
    return finalize_partial_rerun_plan(plan)
