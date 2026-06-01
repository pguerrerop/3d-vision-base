# Template-driven inspection pipelines foundation (RGB POC hardening)

## What changed in this pass

This pass moves the RGB Mining Steel Ball flow from "executable" to "demo-ready usable" without introducing a node editor or replacing the existing artifact/overlay/inspector contracts.

Implemented:

- persisted intermediate RGB pipeline image artifacts per run
- reciprocal overlay <-> measurement object selection support with stable object IDs
- deterministic synthetic RGB demo sample generator script
- tuned RGB template defaults for synthetic demo usability
- manual real-image validation checklist documentation

## Persisted intermediate artifacts

Path: `vision_3d_acquisition/processes/service.py`

For each run, artifacts are now persisted under:

- `data/processes/runs/<pipeline_instance_id>/<run_id>/...`

Persisted images include:

- source RGB image
- ROI image (if enabled)
- grayscale image
- normalized grayscale image
- threshold mask
- morphology mask
- overlay preview image

Behavior:

- no rerun overwrite (run_id-scoped directory)
- run history references persisted paths
- metadata includes:
  - `step_id`
  - `algorithm_key`
  - `source_artifact_id`
  - `image_width`
  - `image_height`
  - `coordinate_space: image_pixel`
  - `persisted: true`

## Object linkage model

Stable object IDs are shared by measurements, classification, and overlays.

Internal linkage:

- numeric `object_id` remains the selection key used by existing architecture

Friendly display convention:

- `object_001`, `object_002`, ...

Path: `frontend/src/components/objectSelectionModel.ts`

## Overlay <-> table reciprocal selection

Implemented via shared `selectedObjectId` state and object mapping helpers.

- selecting contour/ellipse/label updates selected measurement row
- selecting measurement row highlights matching overlays
- inspector/object focus updates via existing selected object flow

Limitation:

- automatic pan/center-to-object in image view is not implemented yet (follow-up)

## RGB demo sample script

Path: `scripts/generate_rgb_steel_ball_samples.py`

Generates deterministic samples in:

- `data/demo/rgb_steel_balls/`

Outputs:

- `one_good_ball.png`
- `multiple_good_balls.png`
- `oval_non_spherical.png`
- `mixed_objects.png`
- `low_contrast_ball.png`
- `metadata.json` with expected counts/classes and notes

This script is manual/dev only; app does not auto-run these samples.

## RGB default parameter assumptions

Path: `vision_3d_acquisition/processes/templates.py`

Defaults are tuned for conservative synthetic usability:

- `rgb_to_gray`: `luminance`
- `normalize_lighting`: `clahe`, moderate clip/tile settings
- threshold: binary with mid-low value baseline
- morphology: close with small kernel to stabilize fragmented masks
- blob min area/circularity: avoids tiny noise while retaining elongated candidates
- ellipse fit/classification confidence thresholds: allow unknown/rejected for low-confidence detections while still separating non-ball/ball on synthetic cases

These defaults are not final production calibration values.

## Manual real-image validation checklist (no automated real-image runs)

1. Capture/select 3-5 real RGB images from the separate camera.
2. Open Processing Lab and create `Mining Steel Ball Classification (RGB/2D MVP)`.
3. Choose `rgb_image` input.
4. Set image input via selected take RGB asset or manual image path.
5. Run process.
6. Inspect artifacts and overlays:
   - source RGB
   - grayscale
   - normalized grayscale
   - threshold mask
   - morphology mask
   - contours/bboxes/centroids
   - ellipse overlays
   - measurement table
   - classifications
7. Tune parameters (ROI, threshold, normalization, blob/ellipse/classification).
8. Rerun and compare run history evidence.
9. Save best parameter configuration.
10. Promote best run to a POC recipe version.

Suggested acceptance notes:

- objects detected correctly
- false positives identified/acceptable for POC
- diameter/roundness plausibility
- rejected/unknown reasons understandable
- overlays align with image evidence
- run history preserves artifacts/measurements/summary

## Remaining limitations

- process-run artifact images are persisted under `data/processes/runs/...`; full first-class rendering in every existing take-centric artifact card path is a follow-up.
- image viewer auto-centering to selected object is pending.
- cross-run comparison UI remains a follow-up.

## Segmentation preview tuning semantics (threshold controls)

Implemented in this pass:

- backend preview endpoint: `POST /api/pipelines/preview-segmentation`
- payload: `take_id`, `pipeline_id`, `params.threshold`, `params.auto_threshold`, `params.invert`
- uses process-service pipeline execution path and real threshold+morphology logic (no frontend-only pixel math)
- preview artifacts include threshold mask, cleaned mask, segmentation overlay, and rejected-components overlay via existing artifact contract

Preview override model:

- preview params temporarily override threshold step behavior only:
  - `mode=otsu` when `auto_threshold=true`
  - `mode=fixed` + `value=<threshold>` when manual mode
  - `invert=<bool>`
- preview does not mutate saved recipe/pipeline instance configuration
- saved recipe state changes only when user clicks `Apply to recipe` or `Apply + rerun`

Frontend state split:

- `persistedParams`: values currently stored in the threshold step config
- `previewParams`: interactive values from slider/toggles
- `dirtyPreviewState`: whether preview params diverge from persisted params

UX behavior:

- slider range: `0..255`
- slider + numeric input are synchronized
- `Auto (Otsu)` disables manual threshold value editing
- preview requests are debounced (`~320ms`) to keep UI responsive and avoid unnecessary full reruns
- `Preview` triggers immediate preview call
- `Apply to recipe` persists threshold params only
- `Apply + rerun` persists params then runs selected pipeline

Future extensibility strategy:

- controls panel is parameter-group oriented (not threshold-only widget logic)
- backend preview contract already accepts structured `params` object and can be extended for:
  - blur kernel
  - morphology operation/kernel
  - min component area
  - ROI toggles
  - adaptive threshold parameters

## Roadmap to production recipe

1. unify process-run artifacts into a richer explorer route with direct image rendering and compare mode
2. add automatic object centering and stronger table/overlay synchronization UX
3. add promotion gates and recipe approval checks
4. add regression compare across recipe versions
5. extend to synchronized 2D/3D/hybrid templates

## Mining Balls Roadmap: Wizard-First Semantic Reconciliation

Mining balls ML-set roadmap now follows a wizard-first workflow:

1. Human-in-the-loop reconciliation of operator tables.
2. Canonical manifest generation with explicit ambiguity diagnostics.
3. Deterministic immutable ML-set artifact materialization.

This keeps template-driven inspection pipelines decoupled from ingestion UX while improving semantic traceability, reproducibility, and future active-learning readiness.
