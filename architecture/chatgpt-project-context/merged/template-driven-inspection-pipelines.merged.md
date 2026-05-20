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

## Note on merged docs generation

No merged-doc generation script was found in repository scripts; merged context was updated manually.
