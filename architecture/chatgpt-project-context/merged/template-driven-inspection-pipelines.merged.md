# Merged context: template-driven inspection pipelines (RGB POC hardening)

## Summary

Implemented additive hardening for the RGB Mining Steel Ball 2D POC:

- persisted intermediate image artifacts per run
- reciprocal overlay/table object selection using stable object IDs
- deterministic synthetic RGB demo sample generator
- tuned RGB defaults for synthetic demo readiness
- manual real-image validation checklist (manual only)

No node editor or workflow graph editor was introduced.

## Persisted artifact behavior

- process runs now persist image artifacts under `data/processes/runs/<instance>/<run>/`
- run history stores artifact paths and metadata
- reruns produce distinct paths (no overwrite)
- persisted metadata includes step and image geometry context (`coordinate_space=image_pixel`)

## Object ID linkage and selection

- selection key remains numeric `object_id` for compatibility
- UI displays friendly IDs (`object_001`, `object_002`, ...)
- overlay selection maps to measurement row selection and vice versa
- inspector focus updates through existing selected-object pathway

## Synthetic demo assets

- `scripts/generate_rgb_steel_ball_samples.py` generates deterministic RGB samples + metadata
- generated set includes good balls, mixed objects, non-spherical example, and low-contrast example
- this is dev/demo tooling only (not auto-executed by app)

## RGB defaults and behavior

- template remains multi-input (`rgb_image`, `grayscale_image`, `reflectance_image`) but defaults to RGB for guided flow
- preprocessing includes ROI crop, RGB-to-gray, and lighting normalization
- threshold/morphology/blob/ellipse/classification defaults tuned for synthetic demo baseline

## Manual real-image validation

Real RGB camera validation is documented as a manual checklist and intentionally not part of automated tests.

## Tests added/updated (synthetic only)

- artifact persistence and rerun path uniqueness
- selection mapping helpers and friendly ID/non-UUID display checks
- demo sample generator determinism and RGB PNG outputs
- artifact-contract compatibility (legacy + projection-aware)

## Remaining limitations

- full direct rendering integration for process-run artifacts across all take-centric explorers is pending
- image auto-center on selected object is pending
- run-to-run compare UX remains pending

## Added: segmentation threshold live preview model

- added backend preview endpoint: `POST /api/pipelines/preview-segmentation`
- endpoint executes real process-service threshold+morphology logic with temporary overrides
- no recipe mutation during preview calls
- preview returns artifact-contract-compatible outputs + diagnostics + segmentation metrics

Parameter semantics:

- preview-only override keys:
  - threshold value (`0..255`)
  - auto threshold (`Otsu`)
  - invert
- threshold mode is derived at preview time:
  - auto => `otsu`
  - manual => `fixed` + provided value

State model in Studio:

- persisted parameters (recipe-backed)
- preview parameters (interactive)
- dirty preview state (preview diverges from persisted)

Execution behavior:

- preview requests are debounced (~320ms) to prevent excessive recomputation
- `Preview` performs an immediate preview call
- `Apply to recipe` persists params only
- `Apply + rerun` persists then runs full pipeline

Extensibility direction:

- current control strip is threshold-focused but structurally ready for additional segmentation knobs
- preview contract remains generic (`params`) for future morphology/ROI/adaptive-threshold live tuning

## Note on merged docs generation

No merged-doc generation script was found in repository scripts; merged context was updated manually.

## Mining Balls Ingestion Flow (Refined)

Mining balls ML-set generation now uses a reusable ingestion wizard for semantic reconciliation prior to deterministic generation.

- Human review remains explicit for unresolved/ambiguous rows.
- Canonical manifest becomes the cross-workflow semantic contract.
- Deterministic artifact generation remains reproducible and immutable-reference based.

## ML Set Governance Surface

- Template-driven inspection workflows now connect to an ML-set governance drawer focused on readiness, balance, reproducibility, and split safety.

## Object-Centric Semantic Governance

- Template-driven workflows now align with a reusable `PhysicalObject` semantic layer for operator reconciliation, repeatability grouping, and object-safe ML splitting.
