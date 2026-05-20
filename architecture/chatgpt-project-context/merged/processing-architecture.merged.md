# Processing Architecture: Family-Aware Status Model

## Why this change

Sensor Studio currently has two active execution systems:

- 3D filesystem pipelines (`data/processed/<take_id>/DONE`)
- 2D ProcessService runs (`data/processes/runs/<pipeline_instance_id>/<run_id>/...`)

A single global `has_done` flag made RGB/2D runs appear unprocessed. This change introduces contextual status by pipeline family while keeping 3D semantics intact.

## Pipeline family model

Normalized family enum:

- `3d`
- `2d`
- `generic`

Pipeline definitions now expose explicit metadata:

- `pipeline_family`
- `execution_backend`
- `supported_modalities`
- `supports_live_processing`
- `supports_batch`
- `supports_partial_stages`

## Take processing summary model

Each take now exposes `processing_by_family` with per-family status:

- `hasCompletedOutput`
- `lastRunAt`
- `status`: `never_processed | completed | running | failed | partial`
- `sources`: `3d_done_marker | process_run`

This allows mixed states on the same take (for example `3d completed`, `2d never_processed`).

## 3D behavior preserved

Unchanged:

- `data/processed/<take_id>/DONE` remains the 3D completion marker
- existing CLIs and live flows remain compatible

## 2D linkage/index model

ProcessService now appends run linkage metadata at execution time into:

- `data/processes/index/runs.json`

Each entry stores:

- `take_id`
- `pipeline_instance_id`
- `run_id`
- `pipeline_family`
- `status`
- `created_at`
- run path

Properties:

- append-only friendly
- resilient if missing
- rebuild fallback from instance histories if index is absent

## Fallback and migration

If the central index is missing, backend rebuilds from persisted pipeline instance execution history when possible.

Older takes remain compatible:

- 3D status still resolved via DONE marker
- 2D status resolved from index/fallback manifests

## Future convergence direction

This model intentionally avoids engine unification now. It enables filesystem queue, ProcessService, and future execution backends to coexist behind one Studio selection/status semantic.

## Explicit pipeline definition model in Studio

Pipeline definitions are now treated as explicit runnable descriptors in UI with metadata fields:

- `id`
- `name`
- `pipeline_family`
- `supported_modalities`
- `execution_backend`
- `supports_live_processing`
- `supports_batch`
- `supports_partial_stages`

This keeps execution engines separate while making run semantics explicit and user-driven.

## Studio selection UX semantics

The Studio UX layer now enforces explicit pipeline selection in the active pipeline card, while preserving backend family-aware status and routing.

No backend engine unification was introduced.

## Studio UI composition ownership

The current Studio page composition explicitly separates concerns:

- Header owns take identity + execution actions.
- Pipeline context card owns pipeline selection/discovery + compatibility.
- Stage section owns stage selection only.
- Trace summary owns run identity/status/timestamp.
- Workspace tabs own result inspection domains (inputs/segmentation/classification/measurements/fusion/artifacts/json).
- Inspector owns selected stage/object/artifact diagnostics.

This avoids previous overlap and duplicated metadata across multiple sections.

## Navigation architecture (stage-first)

UI navigation is explicitly split:

- pipeline structure: stage navigator
- execution state: run summary / execution trace
- result exploration: stage-scoped tabs + artifact views

This preserves backend contracts while preparing for future DAG pipelines:

- current UI renders a linear stage strip
- selected stage is a stable context key
- stage workspace views are resolved from stage outputs, not global categories

Future branching stages can map into the same model by exposing upstream/downstream edges while keeping the `selected stage -> stage workspace` contract unchanged.

## Stage capability registry and renderer abstraction

Frontend now uses explicit semantic configuration for stage visualization:

- `StageSemanticDefinition`: stage category, default view id, view definitions
- `StageViewDefinition`: id/label, renderer type, optional supported artifact semantics, empty states
- renderer dispatch by `rendererType` in `stage_view_renderers` module

This keeps page-level orchestration lightweight while enabling future additions:

- 2D-specific mask/contour/ellipse visualizers
- 3D point-cloud and projection-aware renderers
- multimodal synchronized visualization

The abstraction is additive and does not alter backend execution APIs, filesystem layout, or processing semantics.

## Source artifact resolution hierarchy

Stage renderers now consume a resolved view context with explicit precedence:

- `runArtifacts`: artifacts produced by pipeline execution for selected stage
- `sourceArtifacts`: artifacts adapted from selected take source bindings
- `resolvedArtifacts`: run-first selection with source fallback

Input-stage semantics declare source bindings and therefore remain functional pre-run.
Other stage categories remain run-output dependent.

This is implemented via stage source adapters (`stage_sources`) and preserves renderer abstraction boundaries.

Histogram generation for input-stage visualization is now backend-driven:

- endpoint: `GET /api/takes/{take_id}/source-histogram`
- computation: downsampled luminance histogram
- caching: per-take source cache artifacts under incoming take `.stage_cache`

Frontend consumes the API result and no longer depends on browser canvas image processing for primary histogram behavior.

## Threshold + morphology functional stage (RGB POC)

The first fully functional derived stage is now implemented as `threshold` + `morphology` execution with persisted outputs.

Processing flow:

1. grayscale source (from upstream step)
2. optional blur (`blur_kernel`)
3. threshold (`fixed`, `otsu`, `adaptive`) + optional invert
4. morphology op (`open_close`, `close_open`, `open_only`, `close_only`, `erode`, `dilate`, `none`)
5. optional hole filling
6. optional connected-component cleanup by minimum area
7. optional connected-component max-area filter + ROI zeroing outside segmentation bounds

Generated stage artifacts:

- `threshold_mask` (image)
- `cleaned_mask` (image)
- `overlay_image` (image, mask-over-source blend)
- `morphology_metrics` (metric metadata)
- `morphology_debug_json` (json payload persisted to file)

Contract notes:

- artifacts include `step_id`, `algorithm_key`, `source_artifact_id`
- image artifacts include dimensions + `coordinate_space=image_pixel`
- stage persistence is additive and run-scoped (`data/processes/runs/<instance>/<run>/...`)
- previous run outputs remain immutable (reruns create new run folders)

Segmentation tuning parameters now include:

- threshold: mode/value/invert/blur
- ROI: enabled + x/y/width/height
- morphology: operation/open-kernel/close-kernel/iterations/fill-holes/min-area/max-area
- overlay: alpha + BGR color channels

`morphology_debug_json` now records stage-semantic sections:

- `threshold`
- `morphology`
- `roi`
- `artifacts`

ROI semantics now support both:

- `roi_type=rectangle` with x/y/width/height
- `roi_type=polygon` with `roi_polygon_points` rasterized once per run and applied as ROI mask during threshold/morphology

## Segmentation cleanup generalization

Segmentation cleanup is now generic candidate-mask conditioning, not ball-specific shaping.

## Blob overlay interaction model

Blob/Contour Detection now uses stage-semantic interactive overlay rendering in Studio:

- accepted candidates: thick bright contours, centroid marker, ID label
- rejected candidates: thinner red contours with rejection context
- selected candidate: stronger stroke + glow highlight

Selection contract is reciprocal:

- overlay click selects candidate row + inspector context
- table row click highlights same candidate on overlay

Overlay rendering remains geometry-oriented and generic (no ball-specific logic), and stays decoupled from ellipse/classification stages.

## Ellipse fitting + geometry metrics stage

The next derived stage after blob detection is `ellipse_fitting`, which consumes accepted blob candidates and contour geometry to produce generic fit-quality metrics.

Generated artifacts:

- `ellipse_overlay` (image)
- `ellipse_metrics` (json per-candidate geometry + fit quality)
- `ellipse_summary` (json aggregate)
- `ellipse_debug_overlay` (image diagnostics)

Per-candidate metrics include major/minor axes, equivalent diameter, eccentricity, fill ratio, RMSE/max fit error, circularity, solidity, border touch, and `valid_fit`.

Post-morphology connected-component filters now support:

- `cleanup_min_area`
- `cleanup_max_area`
- `cleanup_min_width`
- `cleanup_min_height`
- `cleanup_max_aspect_ratio`
- `cleanup_border_reject`
- `cleanup_keep_largest_n`
- `fill_holes`

Cleanup diagnostics now include:

- components before/after cleanup
- rejection counts by reason (`small`, `large`, `aspect`, `border`)
- kept component areas
- foreground coverage before/after cleanup

Optional rejected-component visualization is emitted as `rejected_components_overlay`.

## Blob/contour detection stage (real candidate extraction)

`blob_detection` now consumes `cleaned_mask` and emits real candidate artifacts:

- `blob_debug_overlay` (contours + IDs + bbox on source image)
- `blob_labels` (label image)
- `blob_contours` (JSON)
- `blob_metrics` (JSON)
- `blob_rejected` (JSON)

Per-candidate metrics include area, centroid, bbox, perimeter, equivalent diameter, circularity, aspect ratio, solidity, border touch, and parent ROI context.

## Planar 2D camera calibration lane

A dedicated planar RGB calibration lane is now introduced under Calibration UI (`2D Camera`) using OpenCV target detection/calibration APIs.

Capabilities added:

- ChArUco-first, checkerboard-secondary target support
- camera intrinsics estimation
- belt-plane homography estimation for `pixel -> mm`
- persisted `camera_2d` calibration payloads in `config/calibrations`
- ellipse metric conversion to mm when active 2D calibration is present

This is intentionally planar-only and does not yet include depth/height compensation.

## Dataset / Session / Take management layer (MVP)

A new filesystem-backed metadata layer was added for experiment management while preserving all existing processing routes.

- New sidecar root: `data/datasets/`
- No changes to raw take folders in `data/incoming/`
- No changes to run outputs in `data/processed/` and `data/processes/runs/`

`TakeSummary`/`TakeDetail` now include management metadata (friendly name, tags, validation state, expected diameter) plus lightweight run history and thumbnail reference for Studio browsing.

Legacy takes without sidecar metadata are resolved with synthesized defaults at read time, so historical captures remain processable and visible with zero migration.

## 2D calibration source model

The 2D calibration lane now uses a source-discovery contract (`/api/sources`) with explicit freshness status so calibration UI can distinguish live vs stale previews and gate calibration actions accordingly.

## Object annotation boundary (computed vs reviewed)

The processing pipeline remains purely computed output.

- Computed artifacts/metrics continue to live in run outputs.
- Human review metadata is persisted separately in dataset take sidecar metadata (`object_annotations`).

This preserves deterministic processing contracts while enabling supervised validation workflows.

### Candidate association on take detail

At take-detail load time, object annotations are associated to current run candidates using:

1. stable `candidate_id`
2. fallback bbox overlap (IoU)
3. fallback nearest centroid

This association is exposed as metadata (`matched_candidate_id`, `matched_by`) and does not mutate run outputs.

2D calibration now uses explicit capture-first workflow states (`NO_CAPTURES`, `CAPTURES_READY`, `CORNERS_DETECTED`, `CALIBRATION_VALID`) to avoid stale-preview deadlock UX.
