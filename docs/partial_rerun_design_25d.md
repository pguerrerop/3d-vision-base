# Partial Rerun Design for 25D Pipeline

This document defines partial rerun semantics for `mining_steel_ball_classification_25d`.

It is a design milestone first. Sensor Studio does **not** execute partial reruns yet. The goal is to make the safety rules, result shape, graph UX, and compare semantics explicit before any runtime changes are introduced.

## Goals

- preserve existing full-run behavior
- reuse the processing-unit contract as the dependency model
- keep old runs, recipes, comparisons, and graph views compatible
- make future partial reruns safe, inspectable, and explainable

## Non-goals

- no graph editing yet
- no partial rerun execution yet
- no mutation of parent run outputs
- no algorithm rewrite

## Terminology

- **Full run**: normal end-to-end execution of the 25D pipeline.
- **Parent run**: the previous full or partial run whose artifacts may be reused.
- **Partial rerun plan**: a safety-checked description of what could be reused, rerun, or blocked.
- **Stage boundary**: the public stage root such as `remove_belt_segment_objects` or `classification`.
- **Selected unit**: the unit the user clicked in Studio. This can be a substage.
- **Effective rerun boundary**: the public stage boundary used for conservative v1 execution planning.
- **Reusable artifact**: an upstream artifact that is safe to consume without recomputing.
- **Invalidated unit**: a unit whose previous result can no longer be trusted after a parameter or dependency change.

## Supported Modes

| Mode | Meaning | Milestone status |
| --- | --- | --- |
| `full_run` | Execute the whole pipeline from input through overlay. | Existing |
| `rerun_from_unit` | Reuse safe upstream artifacts, rerun selected boundary and downstream. | V1 candidate |
| `rerun_selected_and_downstream` | Same semantic intent as `rerun_from_unit`, but selected from a substage and elevated to a stage boundary. | V1 candidate |
| `run_until_unit` | Stop after a selected unit or boundary. | Future |
| `preview_unit` | Execute a preview-only slice for a unit. | Future |

### Recommended conservative v1

The first executable rollout should support **only rerun from a public stage boundary**, even when the user selects a substage node in the graph.

- selected substage -> planner maps to owning public stage
- selected stage root -> treated as a logical container, not a direct execution target
- unsafe or ambiguous plans -> fall back to `full_run`

This keeps the runtime simple while still giving users meaningful reuse for:

- segmentation tuning
- classification tuning
- overlay regeneration
- later, normalization and detect-reference tuning when upstream compatibility checks pass

## Contract-Based Dependency Model

The dependency model comes from the processing-unit contract, not ad hoc UI logic.

Each unit contributes:

- `id`
- `kind`
- `parent_id`
- `stage_id`
- `order`
- `inputs`
- `outputs`
- `artifacts`
- `controlled_by`
- `supports_partial_rerun`

### Dependency sources

Dependencies should be resolved in this order:

1. **Explicit artifact flow**
   - `outputs[].artifact_id` produced by one unit
   - `inputs[].artifact_id` consumed by another
2. **Declared control flow**
   - `controlled_by` for units that depend on selection/refinement logic without a clean artifact edge
3. **Ordered stage flow**
   - public stages remain strictly ordered:
     - `input`
     - `detect_belt_plane`
     - `normalize_heights_to_plane`
     - `remove_belt_segment_objects`
     - `geometry`
     - `measurement`
     - `measurement_diagnostics`
     - `classification`
     - `overlay`

### Unit classes

Every unit should be classifiable for rerun planning.

| Class | Meaning |
| --- | --- |
| `logical_container` | Stage root or grouping unit; useful for navigation, not a direct execution target. |
| `executable_unit` | A concrete processing step with stable artifacts and/or parameters. |
| `diagnostic_only` | Emits trace or debug metadata but is not a safe standalone execution start. |
| `artifact_emitting` | Produces named artifacts that downstream units can consume. |
| `parameter_only` | Influences behavior but does not directly emit stable artifacts. |
| `optional_source_dependent` | May be absent depending on source modalities, for example reflectance. |

### Practical interpretation for 25D

- public stage roots are **logical containers**
- most substages are **executable units**
- `input.reflectance_rgb` is **optional_source_dependent**
- some summary/diagnostic units are **diagnostic_only**

## Artifact Reuse Semantics

An upstream artifact can be reused only when all of the following hold:

1. Same `take_id`
2. Same source asset identity if available
3. Same calibration snapshot or compatible calibration reference
4. Same processing-unit contract fingerprint
5. Upstream parameters affecting the artifact have not changed
6. Required upstream artifacts are present and readable
7. Upstream trace status is trustworthy

### Compatibility checks

The planner should compare the parent run against current intent using:

- `recipe_snapshot.processing_unit_contract_fingerprint`
- `recipe_snapshot.calibration_snapshot_reference`
- `stage_params` / `parameters_by_unit`
- `processing_unit_trace`
- artifact presence in `result.artifacts`

### Reuse is forbidden when

- contract fingerprint changed
- take or source changed
- calibration changed
- a required artifact is missing
- a required upstream unit failed
- a required upstream unit is already invalidated
- upstream parameters changed
- the selected target is not safe under current v1 policy

## Downstream Invalidation Semantics

Changing a parameter invalidates:

1. the unit that owns the parameter
2. other units in the same effective stage boundary
3. all downstream public stages
4. comparison summaries derived from invalidated units
5. graph badges that previously reflected the parent run state

### Conservative invalidation rule

For v1 planning, if any parameter changes inside a public stage, invalidate:

- the entire owning stage
- every downstream stage

This is intentionally coarse and safe. Finer substage-level execution can come later once more units are explicitly marked rerunnable.

## Planner Output

Future API shape:

```python
plan_partial_rerun(
    pipeline_id,
    selected_unit_id,
    changed_parameters,
    current_recipe_snapshot,
    parent_run_result,
    processing_unit_contracts,
) -> PartialRerunPlan
```

Suggested output:

```json
{
  "safe": true,
  "mode": "rerun_from_unit",
  "selected_unit_id": "classification.primary_heuristic_classifier",
  "selected_stage_id": "classification",
  "effective_start_stage_id": "classification",
  "selection_policy": "stage_boundary_only_v1",
  "stages_to_reuse": [
    "input",
    "detect_belt_plane",
    "normalize_heights_to_plane",
    "remove_belt_segment_objects",
    "geometry",
    "measurement",
    "measurement_diagnostics"
  ],
  "stages_to_rerun": [
    "classification",
    "overlay"
  ],
  "units_to_reuse": ["..."],
  "units_to_rerun": ["..."],
  "units_invalidated": ["..."],
  "required_missing_artifacts": [],
  "warnings": [
    "Selected substage is planned at its public stage boundary."
  ],
  "blocking_reasons": []
}
```

## Partial Run Result Model

Every partial rerun should create a **new run directory**.

Recommended fields:

```json
{
  "run_id": "run_partial_...",
  "parent_run_id": "run_full_...",
  "execution_mode": "rerun_from_unit",
  "start_stage_id": "remove_belt_segment_objects",
  "selected_unit_id": "remove_belt_segment_objects.foreground_thresholding",
  "partial_rerun_plan": {
    "safe": true
  }
}
```

### Reused artifacts

The parent run must never be mutated.

Two safe options exist:

1. copy reused artifacts into the child run output
2. reference parent-run artifacts explicitly and mark them as reused

Recommended first implementation:

- keep a new child run folder
- allow explicit references to parent artifact paths
- normalize those references in the API layer so Studio can still render them

## Trace Schema Extensions

Runtime trace already supports:

- `pending`
- `running`
- `completed`
- `skipped`
- `failed`
- `warning`
- `inferred`

Partial reruns should extend this with:

- `reused`
- `invalidated`

Example:

```json
{
  "processing_unit_trace": {
    "trace_source": "mixed",
    "trace_precision": "mixed",
    "units": {
      "detect_belt_plane.reference_model_fit": {
        "status": "reused",
        "source_run_id": "run_full_001"
      },
      "remove_belt_segment_objects.foreground_thresholding": {
        "status": "completed",
        "trace_source": "runtime_unit_callbacks",
        "trace_precision": "unit_level"
      },
      "classification.primary_heuristic_classifier": {
        "status": "invalidated"
      }
    }
  }
}
```

## Recipe and Override Semantics

- selected recipe provides the baseline
- current edited parameters provide overrides
- result `recipe_snapshot` must store the **full effective recipe**
- a partial run must not store only the changed values
- planner should record which changed parameters caused invalidation

If upstream recipe values differ from the parent run, reuse should be blocked for the affected boundary and downstream.

## Graph UX Semantics

The graph stays read-only.

### Node states for planning

- `reused`
- `rerun`
- `invalidated`
- `blocked`
- `missing_artifact`
- `unsafe_reuse_warning`

### Suggested UI actions

- `Plan rerun from this unit`
- `Run from this unit` later
- `Run until this unit` later
- `Preview this unit` later

For this milestone, only the **planning states** are needed. No graph action must execute pipeline code yet.

## Compare Semantics for Partial Runs

Comparisons must distinguish:

- changed because rerun
- unchanged because reused
- unavailable because outside partial scope

Recommended behavior:

- reused units get an explicit `reused` badge instead of being shown as unchanged
- rerun units retain normal artifact/metric/classification diff behavior
- summary cards should count reused units separately from changed units

## Recommended V1 Implementation Scope

1. planner only, no execution
2. stage-boundary-only reuse policy
3. selected substage maps to owning public stage
4. stage roots remain logical containers, not direct execution targets
5. new child run result required for any future partial execution
6. fallback to full rerun on any unsafe condition

## Risks and Open Questions

- some same-stage dependencies are still encoded more strongly by order and `controlled_by` than by pure artifact lineage
- a few optional/source-dependent units need explicit non-blocking treatment
- reused artifact referencing needs a stable API strategy before execution rollout
- compare and graph UIs will need dedicated badges for `reused` and `invalidated`
- later substage-level execution should only be enabled for units explicitly marked safe in the registry

## Lightweight Validation Helper

This milestone includes a planning-only helper in:

`vision_3d_acquisition/pipelines/partial_rerun_plan.py`

It does not execute pipeline code. It validates:

- contract fingerprint compatibility
- missing required artifacts
- conservative downstream invalidation
- stage-boundary-only planning

That helper is intended to make the design concrete and testable without changing runtime behavior.
