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

## Acquisition-centric vs classification-centric rendering intent

Studio now applies context-specific rendering semantics without introducing separate apps or pipeline forks:

- acquisition-centric for take browsing/curation (full-frame previews, session/take semantic chips, dataset/session context prominence)
- classification-centric for runtime/stage QA overlays and diagnostics

The change is visual/semantic only and preserves:

- stage-first navigation
- artifact-first contracts
- immutable take identity
- many-runs-per-take processing model

Context semantics are explicit:

- selected context resolves from selected take metadata/summary
- filter state remains separate browsing intent

This avoids semantic confusion between “what I am inspecting” and “what list constraints are active.”

Visual hierarchy refinements remain non-architectural:

- control weighting: acquisition identity -> pipeline -> stage -> actions
- runtime card density tuning: larger full-frame previews with preserved operational class/status emphasis
- inspector compaction: reduced whitespace, unchanged diagnostic content

Performance boundary is explicit:

- sidebar list loading is summary-only and paginated
- selected detail loading remains independent and lazy
- stage/artifact hydration stays selected-take scoped

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

## Classification overlay artifacts (first-class)

Classification visualization now persists as explicit artifacts, not ephemeral UI drawings:
- `classification_overlay.png` -> artifact id `classification_overlay_image` (`kind: "image"`, stage `classification`)
- `classification_overlay_metadata.json` -> artifact id `classification_overlay_metadata` (`kind: "json"`, stage `classification`)

Overlay metadata contract:
- `artifact_kind: "overlay"`
- `overlay_type: "classification"`
- `target_artifact_id: "source_rgb_image"`
- `overlay_coordinate_space: "image_pixel"`
- `objects: [...]` where each object includes:
  - `object_id` (stable friendly id, example `object_007`)
  - `source_object_id` (numeric object id for UI selection sync)
  - `label`, `confidence`, `status`
  - `contour`, `bounding_box`, optional `ellipse`
  - `measurements`, `annotations`

Why persisted artifacts:
- report/export/dashboard flows can consume overlay evidence without recreating UI logic
- replay/debug pipelines can inspect the same overlay payload used in Studio
- contract consistency is preserved across API consumers and future analytics tools

Studio classification behavior:
- default view is now `Overlay`
- overlay click selects object row/card
- row/card select highlights overlay object
- lightweight controls toggle IDs, labels, confidence, contours, and geometry overlays
- legacy runs without classification overlay artifacts stay functional with graceful fallback views

## 2.5D classification explanation artifact contract

2.5D classification emits a canonical auditable artifact:

- file: `classification_explanation.json`
- artifact id: `classification_explanation`
- kind: `json`
- stage: `classification`
- scope: `global`

Payload contract:

- top-level `objects[]` per classified object
- each object explanation includes:
  - `object_id`
  - `final_class_name`
  - `final_class_label`
  - `superclass`
  - `subclass`
  - `confidence`
  - `decision_summary`
  - `metrics_used`
  - `rules[]`
- each `rules[]` entry includes:
  - `rule_id`, `label`, `description`
  - `metric_key`, `value`
  - `expected_range` or `threshold`
  - `comparator`
  - `passed`
  - `severity` (`info|warning|critical`)
  - `contribution` (`positive|negative|neutral`)
  - `message`
- artifact metadata includes:
  - `semantic_type: "classification_explanation"`
  - `artifact_type: "classification_explanation"`
  - `scope: "global"`

Per-object references:

- each object includes:
  - `classification_explanation_ref.artifact_id = "classification_explanation"`
  - `classification_explanation_ref.object_id = <object_id>`

Diameter provenance is explicit:

- `diameter_ellipse_mm`
- `diameter_equivalent_area_mm`
- `diameter_circumference_mm`
- `diameter_selected_mm`
- `diameter_selected_source`
- `diameter_sanity_status`
- `diameter_sanity_message`

Sanity rules explicitly surface suspicious or impossible conditions:

- diameter vs `dim_x/dim_y` inconsistencies
- missing footprint area
- invalid/unstable ellipse fit evidence
- suspicious scale factors
- extremely low roundness/sphericity

Known-cube correction is authoritative for measurement/classification:

- saved or measured scale factors are applied before classification consumes object metrics
- correction context is persisted in measurement metadata:
  - `correction_source`
  - `scale_x`, `scale_y`, `scale_z`
  - `correction_applied`
  - `correction_context_id`
- objects include:
  - `metrics_coordinate_space` (`corrected_mm|raw_mm`)
  - `correction_source`
  - `correction_context_id`
- rule explanations include raw/corrected metric values and correction scale context

Canonical corrected geometry space:

- `geometry_coordinate_space: corrected_metric_mm`
- corrected per-object geometry representations:
  - `contour_mm`
  - `ellipse_fit_mm`
  - `covariance_axes_mm`
  - `point_cloud_mm` (center/height context)
- shape descriptors (eccentricity, roundness/circularity/sphericity, equivalent/circumference diameters) are computed from corrected metric-space geometry.
- `feature_sphericity_3d` is the canonical `min(dim_x, dim_y, dim_z) / max(dim_x, dim_y, dim_z)` over corrected XYZ extents (`dimensions_mm`), recomputed after any contour-based override so it never inherits a stretched / misaligned raw ellipse semi-axis.
- `dim_z` (i.e. `dimensions_mm[2]`) is sourced from the **P99** of `height_above_belt`, NOT the absolute max. This applies to every object in the pipeline and to the known-cube calibration measurement, so a few sharp noise peaks on the cube surface (or on regular objects) do not inflate the canonical Z extent. The absolute peak is preserved separately under `height_above_belt_mm.max_height_mm` (and under `known_object_scale_validation.measured_height_max_mm` for the cube) for diagnostics. The cube validation result also exposes `measured_height_source = "p99_height_mm"` so the studio UI can explain where dim_z came from.
- `feature_local_curvature_proxy` is computed from per-axis gradient components (`feature_curvature_components_raw.mean_abs_g{x,y}_mm_per_mm`) at measurement time over the inner-eroded mask, then rescaled to corrected metric space via the per-axis factors `scale_z/scale_x` and `scale_z/scale_y`.
- raw/corrected values are preserved for engineering diagnostics.

Geometry-debug workspace (engineering):

- measurement stage exposes `Geometry debug` view with per-object geometry debug artifacts and summary.
- inspector surfaces raw vs corrected eccentricity/roundness/diameter and correction scales.
- geometry-debug artifacts are emitted by both the live known-cube calibration path and the persisted-known-cube reuse path.

Invariant validation rules:

- `invariant_violation:near_equal_dims_with_extreme_eccentricity` (still severity=warning, contribution=negative)
- `invariant_violation:diameter_far_exceeds_dims`
- `invariant_violation:compact_blob_with_near_zero_roundness`
- `warning:anisotropic_scale_factors_may_destabilize_metrics` is only raised when the contour-based corrected geometry could NOT be derived; once a valid corrected contour exists, the metrics live in true mm space and the advisory is suppressed. When raised, it is informational (severity=info, contribution=neutral).

Corrected-metric classification semantics:

- `sanity.scale_factors` is informational once a corrected metric geometry exists, so calibrated anisotropic scales (e.g. TriSpector `scale_y=0.2`, `scale_z=0.0365`) no longer fail the rule.
- `shape.deformation_score` is demoted to informational when the primary shape consensus (`feature_sphericity_3d >= 0.8`, `sphericity_score >= 0.75`, `feature_eccentricity < 0.45`) already supports a ball.
- Per-rule `metric_source` flips to `corrected_metrics` once `metrics_coordinate_space == corrected_mm`.

## Studio Rule Explanation View (Engineering)

Classification stage includes a dedicated `Rule explanation` tab:

- compact object selector (multi-object runs)
- final decision card (class/label/superclass/subclass/confidence)
- rule table ordered as:
  - critical failed rules first
  - warning/sanity rules next
  - passed supporting rules last
- columns: Rule, Value, Expected/threshold, Pass/fail, Effect, Message
- resolver order:
  - artifact metadata semantic type
  - artifact id alias (`classification_explanation`, `classification_explanation_json`, `classification_explanation_metadata`)
  - filename fallback (`classification_explanation.json`)
- empty state includes a debug artifact list (id/kind/path) for classification-stage artifacts.

Inspector adds a compact `Why this class?` section with top decisive rules for selected object.

Metric-level explainability contract:

- artifact: `metric_explanation.json`
- artifact id: `metric_explanation`
- stage: `classification`
- kind: `json`
- scope: `global`
- object payload includes `metric_trace[]` entries with:
  - `trace_id` (object-scoped stable id, e.g. `obj_3_5`)
  - `metric_key`
  - `final_value`
  - `raw_value`
  - `corrected_value`
  - `correction_applied`
  - `correction_scales` (`{x, y, z}` from `scale_correction_applied`)
  - `correction_factor_used` (per-metric scalar actually applied; e.g. `scale_z` for `max_height_mm`, `scale_x*scale_y` for `footprint_area_mm2`, `scale_x*scale_y*scale_z` for `feature_volume_proxy_mm3`; null for derived dimensionless metrics)
  - `coordinate_space_before` (e.g. `raw_metric_mm`)
  - `coordinate_space_after` (e.g. `corrected_metric_mm`)
  - `geometry_metric_source` (`contour_corrected` when a corrected mm contour drives planar metrics, `bbox_corrected` when only bbox extents were rescaled, `raw_measurement` when no correction was applied)
  - `source_artifact_id` (`geometry_debug_summary` or `measurement_object` or `known_object_scale_validation`)
  - `source_stage`
  - `formula_name`
  - `formula_human_readable` (math-style expression with units)
  - `formula_inputs` (`[{name, value}, ...]`)
  - `intermediate_values` (free-form dictionary, e.g. axis ratios, scale products, gradient components, `geometry_metric_source`, `geometry_coordinate_space`, `diameter_selected_source`, `diameter_sanity_status`)
  - `validity_status` (`valid|suspicious|invalid|not_applied`)
  - `warnings` (e.g. `invariant_violation:*`, `warning:*`, diameter sanity messages)
  - `used_by_classifier` (true when the same `metric_key` appears in `classification_explanation.rules[]`)

Per-object `metric_trace[]` always includes (when the data exists on the object):

- `feature_eccentricity`, `sphericity_score`, `feature_sphericity_3d`
- `diameter_selected_mm`
- `feature_local_curvature_proxy` (with per-axis gradient components `mean_abs_gx_mm_per_mm`, `mean_abs_gy_mm_per_mm` in `intermediate_values`, plus `scale_z/scale_x`, `scale_z/scale_y` and `computed_over` reflecting the inner-eroded mask used for the gradient sample)
- `max_height_mm`, `footprint_area_mm2`, `feature_volume_proxy_mm3`
- `scale_correction_applied` (synthetic calibration context entry capturing `correction_source`, `correction_context_id`, scales and the resulting `geometry_coordinate_space`)
- `geometry_invariant_warnings` (synthetic entry, only when invariants triggered, split into `violations` and `advisories` in `intermediate_values`)

Studio engineering view:

- classification stage exposes a `Metric details` tab implemented by `MetricDetailsPanel`.
- the tab renders a `Correction context` card at the top showing `correction_source`, `correction_context_id`, applied scales, `geometry_metric_source` and `geometry_coordinate_space`, plus an `Invariants` section that separates `invariant_violation:*` entries from `warning:*` advisories.
- the metric table shows raw vs corrected value, per-metric `correction_factor_used`, coordinate-space transition, source artifact, formula name + math expression, and validity badge (`status-pill ok|warn|bad`).
- each row is expandable to reveal `formula_inputs`, `intermediate_values`, and per-metric `warnings`.
- rule explanation rows include `metric_trace_ref` so rule metric values can be traced back to the same `trace_id`.

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

Corner-detection hardening now includes dictionary-aware ChArUco detection metadata (dictionary name, marker IDs, API mode, sharpness estimate, board coverage, failure reason) and selected-capture-first detection semantics.

Runtime tuning for calibration adds camera-control diagnostics (exposure/focus/gain family) and image-quality telemetry (brightness histogram, clipping, sharpness) to improve marker detection consistency before calibration solve.

UI architecture now separates runtime tuning from calibration solving via dedicated modals, reducing overloaded page state and keeping calibration persistence concerns independent from live stream concerns.

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

## Capture-first calibration state machine

2D calibration workflow exposes explicit states:

- `NO_CAPTURES`
- `CAPTURES_READY`
- `CORNERS_DETECTED`
- `CALIBRATION_VALID`

Detect corners is gated by capture existence + valid target parameters.
Calibrate is gated by successful corner detection.
Preview stale state does not gate calibration progression.

## ProcessBinding + immutable recipe runtime (step 1)

Studio remains the editor/debugger for process templates and recipes, but runtime execution now resolves an explicit `ProcessBinding` before running when available.

- Binding key: `source_id + modality + purpose`
- Binding target: `pipeline_id + active_recipe_version_id + optional calibration_profile_id`
- Purposes: `acquisition_inspection`, `manual_debug`, `fusion_input`, `fusion`

Runtime execution uses immutable recipe-version snapshots for process-service pipelines: workers/manual runs execute the bound recipe snapshot instead of mutable live instance config.

`PipelineRunner`/single backend execution path is now shared between manual Studio-triggered runs and worker-style processing entrypoints.

Run metadata persistence now records:

- `pipeline_id`
- `recipe_version_id`
- `config_snapshot_hash`
- `source_id`
- `take_id`
- `acquisition_group_id` (nullable)
- `calibration_profile_id` (nullable)

Fusion behavior is intentionally not implemented in this step; `fusion_input`/`fusion` purposes are reserved binding semantics only.

## Acquisition Process Split: 2D vs 2.5D

- Acquisition is a source/process layer and is separate from classification/inspection stages.
- 2D USB capture is treated as acquisition source output (`source_id=usb_camera_<index>`, modality `rgb`).
- 2.5D TriSpector ingest is treated as acquisition source output (`source_id=trispector_ftp_<n>`, modality `heightmap`).
- Both flows publish raw/normalized assets to `incoming/<take_id>/` through the shared acquisition publisher contract (`metadata.json` + `READY`).

## TriSpector FTP Ingestion Contract

- FTP upload ingestion only performs:
  - file stability/ingest,
  - TriSpector 2.5D parsing,
  - persistence of parsed assets and parser diagnostics,
  - take registration with modality metadata.
- Ingestion does **not** run classification directly in the FTP handler.
- Parser output contract for registered takes includes:
  - `height16.tif` as `files.heightmap`,
  - `reflectance.png` when available as `files.reflectance`,
  - `parser_metadata.json` as `files.parser_metadata`,
  - `heightmap_preview.png` as `files.heightmap_preview`,
  - original uploaded file as `files.raw_upload`.

## Binding-Driven Processing Trigger After Acquisition

- After a take is registered, acquisition resolves bindings by:
  - `source_id`,
  - `modality`,
  - `purpose` (default `acquisition_inspection` for runtime ingestion).
- If an active binding exists, the bound immutable recipe version execution path is triggered.
- If no binding exists, ingestion succeeds and the take remains `READY` with warning:
  - `No active processing binding found for this source/modality/purpose.`
- No modality-cross fallback is allowed for 2.5D bindings.

## Studio and Runtime Alignment

- Studio/manual processing and runtime acquisition-triggered processing both resolve through the same `ProcessBindingService` key (`source_id + modality + purpose`).
- This keeps source-to-recipe routing consistent across interactive and unattended workflows.

## Acquisition-to-Processing Seam (read-only contract)

Ownership split:

- Acquisition owns capture/ingest/parsing/upload/raw persistence/take creation.
- Processing owns binding resolution + pipeline execution through the existing runner path.

Shared contract: `AcquiredTakeReady`

- `take_id`
- `source_id`
- `modality`
- `modality_family` (optional)
- `acquisition_process_id` (optional)
- `acquisition_run_id` (optional)
- `acquisition_group_id` (optional)
- `session_id` (optional)
- `asset_paths`
- `metadata`
- `warnings`
- `created_at`

Modality normalization rule:

- Canonical processing modality is `rgb` for 2D and `heightmap` for 2.5D.
- Alias `heightmap_2_5d` is normalized to `heightmap` for binding resolution.
- Original alias is preserved at `metadata.original_modality`.

Resolution service:

- `process_acquired_take_if_bound(acquired_take, purpose=\"acquisition_inspection\", auto_process=True)`
- If `auto_process=False`: status `ready_not_processed`.
- If no binding: status `processing_binding_missing` with warning (non-fatal).
- If binding exists: dispatch via shared processing path (`dispatch_take_processing`) with bound:
  - `pipeline_id`
  - `active_recipe_version_id`
  - `calibration_profile_id` (optional)
- No modality-cross fallback and no direct FTP-handler classification trigger.

Status lifecycle (`data/acquisition_processing/status/<take_id>.json`):

- `acquired`
- `ready_for_processing`
- `processing_binding_missing`
- `processing_enqueued`
- `processing_running`
- `processing_completed`
- `processing_failed`
- `ready_not_processed`

Persisted run metadata includes seam context:

- `take_id`, `source_id`
- `acquisition_group_id`, `acquisition_process_id`, `acquisition_run_id`
- `pipeline_id`, `recipe_version_id`, `config_snapshot_hash`
- `modality`, `modality_family`

API seam endpoints:

- `POST /api/acquisition/processing/resolve`
- `GET /api/takes/{take_id}/acquisition-processing-status`

Studio visibility:

- Take summary/detail include `acquisition_processing_status`.
- This is rendered in existing take-detail/Studio flows, without a separate acquisition UI implementation path.

Future fusion dependency:

- `acquisition_group_id` remains the grouping key for future RGB+25D fusion stages.

## Fusion preparation layer (RGB + 25D, group-scoped)

This step adds a read-only fusion readiness resolver over existing processed outputs, keyed by `acquisition_group_id`.

### Fusion input bundle contract

`FusionInputBundle`:

- `acquisition_group_id`
- `rgb_take_id` / `heightmap_take_id` (optional)
- `rgb_run_id` / `heightmap_run_id` (optional)
- `rgb_result_payload` / `heightmap_result_payload` (optional)
- `rgb_artifacts` / `heightmap_artifacts`
- `readiness_status`
- `missing_inputs`
- `warnings`
- `created_at` / `resolved_at`

Readiness lifecycle:

- `waiting_for_rgb`
- `waiting_for_heightmap`
- `waiting_for_processing`
- `ready_for_fusion`
- `incomplete`
- `failed_input`

Rules:

- Resolver consumes existing takes/runs only.
- No acquisition triggering, no FTP coupling, no forced processing mutation.
- Latest successful RGB/25D run is selected by default.
- Multiple candidates are surfaced in warnings + debug candidate list.

### Preliminary object-level alignment

`FusionObjectCandidate` is generated for debug pairing with tolerant, non-fatal matching:

- `centroid_projection`
- `bbox_overlap`
- `object_index_fallback` (current MVP default when transforms are absent)
- `unmatched_rgb`
- `unmatched_heightmap`

Missing optional fields (bbox/centroid/class/confidence) emit warnings instead of failing.

### Preview payload + APIs

`FusionPreviewResult` returns:

- readiness status
- object pairing candidates
- merged feature summary
- recommended next action
- warnings

APIs:

- `GET /api/acquisition-groups/{group_id}/fusion-inputs`
- `GET /api/acquisition-groups/{group_id}/fusion-preview`

### Studio debug visibility

When a selected take has `acquisition_group_id`, Studio shows fusion-readiness status, selected run IDs, missing inputs, and a compact object pairing table.

### Current limitation and future path

- Current pairing is debug-oriented and may fall back to index pairing.
- Calibration-based geometric matching/projection is a future step.
- Operator-facing fused classification/publication is intentionally deferred.

## Persisted fusion pipeline/run (RGB + 25D)

New pipeline definition:

- `pipeline_id = mining_steel_ball_fusion_rgb_25d`
- `pipeline_family = fusion`
- `execution_backend = native`
- group-scoped execution keyed by `acquisition_group_id`

### Fusion run contract

`FusionRun` persisted under:

- `data/fusion/runs/group_<acquisition_group_id>/run_<fusion_run_id>/`

Fields:

- `fusion_run_id`
- `acquisition_group_id`
- `pipeline_id`
- `pipeline_family = fusion`
- `recipe_version_id` (optional)
- `config_snapshot_hash` (optional/configurable)
- `rgb_take_id`, `heightmap_take_id`
- `rgb_run_id`, `heightmap_run_id`
- `status`, `started_at`, `completed_at`
- `result_path`
- `artifact_paths`
- `warnings`

### Fusion result contract

`FusionClassificationResult` persisted in `fusion_result.json` with:

- `acquisition_group_id`
- `fusion_run_id`
- `status`
- `final_objects`
- `class_counts`
- `warnings`
- `source_refs`
- `artifact_refs`

Each `FinalInspectionObject` includes:

- `final_object_id`
- `rgb_object_id` / `heightmap_object_id`
- `final_class`
- `final_class_group`
- `confidence`
- `decision_reasons`
- `rgb_evidence`
- `heightmap_evidence`
- `measurements`
- `matching_method`
- `matching_confidence`

### Rule-based MVP classification

Initial fusion rules (configurable thresholds):

- 25D scrap-like elongated/flat evidence -> `Chatarra`
- high 25D deformation -> `Scrap de Bola / Bola deformada`
- RGB chip/crack hints -> `Scrap de Bola / Bola con chip` or `Bola partida`
- healthy round consensus -> `Bola buena`

Decision reasons are always emitted for traceability.

### Persisted fusion artifacts

- `fusion_result.json`
- `fusion_summary.json`
- `fusion_object_table.json`
- `fusion_debug_pairing.json`

Optional overlay generation is deferred; failure to generate overlay should not fail fusion run.

### Fusion APIs

- `POST /api/acquisition-groups/{group_id}/fusion-runs`
- `GET /api/acquisition-groups/{group_id}/fusion-runs`
- `GET /api/fusion-runs/{fusion_run_id}`
- `GET /api/acquisition-groups/{group_id}/fusion-result/latest`

Readiness gate:

- POST requires `ready_for_fusion` by default.
- `force=true` allows execution with incomplete readiness.

### Why operator publication remains deferred

This step focuses on persisted, reviewable fusion evidence and reproducible debug outputs. Operator publication requires a separate acceptance contract, station-level fallback behavior, and production confidence policy.

### Future recipe-bound execution

Fusion run metadata already carries `recipe_version_id` and `config_snapshot_hash`, preparing later binding semantics:

- station/source + `purpose=fusion`
- `mining_steel_ball_fusion_rgb_25d`
- immutable fusion recipe/config snapshots

## AcquisitionGroup for multimodal capture grouping (step 2)

`acquisition_group_id` is now the synchronization primitive for grouping captures from the same physical conveyor event across 2D and 2.5D/3D lanes.

`AcquisitionGroup` includes:

- `id`
- `name` (nullable)
- `station_id` (nullable)
- `trigger_id` (nullable)
- `encoder_position` (nullable)
- `started_at`
- `completed_at` (nullable)
- `status` (`open | complete | failed`)
- `metadata` (dict)

Grouping can be created and attached manually first. Capture APIs now accept optional `acquisition_group_id` and propagate it into take metadata and pipeline run metadata.

This `AcquisitionGroup` boundary is the future fusion boundary. Automatic grouping strategies (trigger/encoder/timestamp window) are planned later. Fusion execution remains intentionally deferred.

## ObjectCandidate semantic contract for pre-fusion pipelines (step 3)

2D and 2.5D pipelines now emit a shared object-level semantic contract under `result_payload.object_candidates` while preserving all legacy fields (`objects`, artifacts, diagnostics).

`ObjectCandidate` is the inter-pipeline semantic output boundary and includes identity, source provenance, image/world localization, geometry/appearance/measurement dictionaries, classification hints, and diagnostics.

- 2D pipelines populate candidates from contour/blob + ellipse + classification artifacts.
- 2.5D pipelines populate candidates from connected components, height metrics, and classification overlays.
- Empty detections emit `object_candidates: []` (field present, not omitted).

Fusion is intentionally not implemented in this step. Future fusion will consume `ObjectCandidate[]` from each modality, while pipelines continue to expose modality-specific artifacts and diagnostics.

## Fusion pipeline over ObjectCandidates (step 4)

A first executable fusion pipeline is now available as `mining_steel_ball_fusion`.

- Input boundary: `ObjectCandidate[]` emitted independently by 2D and 2.5D pipelines.
- Grouping boundary: `acquisition_group_id`.
- Output contracts: `FusionResult` and `FinalObject`.

Initial matching is pixel-space POC only:

- centroid distance threshold first
- optional bbox IoU fallback
- unmatched candidates preserved

Initial classification is transparent/rule-based with explicit `decision_reasons`, combining strong 2D hints with 2.5D deformation/shape/height hints.

World/calibrated matching is intentionally deferred.

## PublishedInspectionResult operator contract (step 5)

Operator-facing consumption now uses a dedicated publication boundary, separate from fusion/debug internals.

- Contract: `PublishedInspectionResult`
- Publication service: `publish_fusion_result(data_dir, fusion_run_id, station_id=None, persist=True)`
- Persistence root: `data/published/inspection_results`
- Records: `data/published/inspection_results/<published_result_id>/published_result.json`
- Display summary: `data/published/inspection_results/<published_result_id>/display_summary.json`
- Index + pointers: `data/published/inspection_results/index.json`

`PublishedInspectionResult` fields:

- `published_result_id`
- `acquisition_group_id`
- `fusion_run_id`
- `station_id` (nullable)
- `session_id` (nullable)
- `timestamp`
- `status`: `pending | complete | incomplete | failed`
- `overall_decision`: `accept | reject | review | unknown`
- `primary_class`
- `primary_class_group`
- `confidence`
- `class_counts`
- `objects: PublishedInspectionObject[]`
- `warnings[]`
- `display_artifacts`
- `source_refs`

`PublishedInspectionObject` includes:

- `final_object_id`
- `final_class`
- `final_class_group`
- `confidence`
- `decision_reasons`
- `key_measurements`
- `display_label`
- `sort_order`

Overall decision mapping:

- only `Bola buena` objects -> `accept`
- any `Scrap de Bola` or `Chatarra` class/group -> `reject`
- warnings or low confidence -> `review`
- failed/missing/empty result -> `unknown`

Display artifact fallback:

- preferred main overlay: fusion overlay reference
- fallback: RGB overlay reference
- fallback: 2.5D/heightmap overlay reference
- fallback: none (publication still succeeds)

API surface for Operator UI:

- `POST /api/acquisition-groups/{group_id}/publish`
- `POST /api/acquisition-groups/{group_id}/fuse-and-publish`
- `POST /api/acquisition-groups/{group_id}/fusion-runs` with `auto_publish` (optional)
- `GET /api/operator/inspection-results/latest?station_id=...` (optional filter)
- `GET /api/operator/inspection-results/{published_result_id}`
- `GET /api/operator/inspection-results?session_id=...&station_id=...&limit=...`

Boundary rule:

- Studio/debug contracts (`FusionResult`, run artifacts, pairing tables) remain engineering/debug surfaces.
- Operator APIs expose only `PublishedInspectionResult` shape.
- Publication stores references to source/fusion artifacts; it does not duplicate large debug artifacts.

## Minimal Operator UI (step 6)

First operator-facing UI is intentionally small and read-only:

- Route: `/operator` and `/operator/inspection`
- Page: `OperatorInspectionPage`
- Data source: operator publication APIs only
  - `GET /api/operator/inspection-results/latest`
  - `GET /api/operator/inspection-results/{published_result_id}`
  - `GET /api/operator/inspection-results?limit=20&station_id=...`

UI behavior:

- dominant decision card (`ACCEPT | REJECT | REVIEW | UNKNOWN`)
- class counts and compact object table
- visual evidence via `display_artifacts.main_overlay` when resolvable
- fallback text when no display image is available
- recent results list allows loading a selected published result
- manual refresh plus default 3s auto-refresh

Boundary constraints:

- does not use studio pipeline status, acquisition controls, or debug payload internals
- does not expose raw FTP/debug internals as operator-facing labels
- currently trusts publication-layer artifact references and only renders resolvable API/URL paths

## Local Runtime Process Supervisor foundation

Sensor Studio now includes a generalized local runtime/process supervision layer for long-running operator processes.

Core runtime model:

- `RuntimeProcessDefinition`: launch contract (`process_id`, process type, command, source context, config).
- `RuntimeProcessInstance`: live process state (`pid`, status, lifecycle timestamps, restart count, health, heartbeat, last event summary).
- `RuntimeProcessEvent`: structured operational events (`timestamp`, `severity`, `event_type`, `message`, `metadata`).

Supervisor responsibilities:

- launch and monitor local subprocesses (`subprocess.Popen`)
- track process lifecycle (`start`, `stop`, `restart`, `status`, `list`)
- capture `stdout/stderr`, expose tail logs, and emit structured runtime events
- support graceful shutdown for managed local runtimes

First managed runtime process:

- `trispector_ftp` (type `trispector_ftp_runtime`) runs under the supervisor.
- FTP server command is configurable from `config/runtime.json`.
- Upload lifecycle folders are explicit: `incoming/`, `processing/`, `processed/`, `failed/`.
- File stability gate requires non-zero size and unchanged size across N checks.
- Stable uploads are parsed/registered through existing `TriSpectorFtpAcquisitionAdapter`; bound processing dispatch remains in acquisition-processing integration.

Operator/API/CLI surface:

- `GET /api/runtime/processes`
- `GET /api/runtime/processes/{process_id}`
- `POST /api/runtime/processes/{process_id}/start`
- `POST /api/runtime/processes/{process_id}/stop`
- `POST /api/runtime/processes/{process_id}/restart`
- `GET /api/runtime/processes/{process_id}/logs`
- `GET /api/runtime/processes/{process_id}/events`
- `python scripts/runtime.py list|start|stop|restart|logs`
- `python scripts/runtime.py start trispector_ftp --foreground`

Runtime execution hardening (venv/macOS/Linux safe):

- Runtime subprocesses resolve their interpreter canonically from `sys.executable` first, with explicit fallback warnings when `python3`/`python` must be used.
- FTP commands configured as `python -m ...` or `python3 -m ...` are normalized to the resolved interpreter automatically.
- For `pyftpdlib`, supervisor/runtime enforce `-d <upload_dir>` injection when missing and emit `FTP_ROOT_MISMATCH` when configured command root differs from watched upload root.
- Startup preflight validates executable presence, upload dir existence/writability, FTP port availability, and `pyftpdlib` importability before the process is considered healthy.
- Validation and normalization events are explicit: `PROCESS_VALIDATION_FAILED`, `FTP_COMMAND_NORMALIZED`, and `FTP_ROOT_INJECTED`.
- Process recovery is persistent across short-lived CLI/API invocations using PID records + persisted process metadata + live process inspection, so `list` can report recovered running state without in-memory continuity.
- Ownership checks include process id, command signature, cwd, and persisted metadata to reduce false `STALE_PID_OWNERSHIP_MISMATCH` warnings.

Conceptual boundary (important):

- Pipelines process data.
- Runtime processes produce/watch/execute work.
- Bindings connect acquisition and processing.

This runtime layer stays intentionally lightweight and local-only (no distributed orchestrator) while providing a reusable foundation for future filesystem watchers, preview loops, workers, Ruler streaming acquisition, and live inference services.

## Supervised runtime workers (step 6)

The first supervised runtime worker layer enables automatic operation outside Studio while reusing the same bindings, recipes, pipeline execution, fusion, and publication service paths.

Worker contracts:

- `WorkerDefinition`
- `WorkerStatus`
- `WorkerHeartbeat`
- `WorkerEvent`
- `WorkerRunSummary`

Worker manager/service responsibilities:

- `register_worker(...)`
- `get_worker_status(worker_id)`
- `list_worker_statuses()`
- `append_worker_event(worker_id, event)`
- `heartbeat(worker_id, status)`
- `mark_worker_error(worker_id, error)`

POC workers:

- RGB acquisition-processing worker
- 2.5D acquisition-processing worker
- fusion publisher worker

Execution rules:

- workers do not duplicate Studio/editor logic
- workers call existing backend service paths
- workers use active `ProcessBinding` entries only
- worker failures emit diagnostics/events instead of crashing the runtime loop
- workers support `--once` and `--dry-run` for safe testability

Minimal grouping strategy (current step):

- use existing `acquisition_group_id` when present
- otherwise create one acquisition group per take
- fusion worker fuses only when a group has at least one processed 2D candidate result and one processed 2.5D candidate result
- advanced trigger/encoder grouping remains deferred

Worker monitoring API (minimal by design):

- `GET /api/runtime/workers`
- `GET /api/runtime/workers/{worker_id}`
- `GET /api/runtime/workers/{worker_id}/events`
- `POST /api/runtime/workers/{worker_id}/stop-request`

CLI runners:

- `scripts/run_rgb_worker.py`
- `scripts/run_25d_worker.py`
- `scripts/run_fusion_publisher_worker.py`

## Studio runtime setup/status UI (step 7)

Studio now exposes a compact runtime supervision view for engineering operations without becoming the Operator UI.

Scope of Studio runtime UI:

- Runtime panel lists registered workers with state, heartbeat, source/station context, and latest event summary.
- Worker detail shows recent events/diagnostics and supports stop-request only.
- Processing Lab shows compact runtime health badges for RGB worker, 2.5D worker, and fusion publisher worker.
- Processing Lab shows latest published inspection result id/status as runtime context only.

## Lightweight live orchestration layer

Runtime keeps process supervision and now adds worker-oriented orchestration on top of existing binding and pipeline execution contracts.

### Worker model

- Worker types: `acquisition`, `processing`, `fusion`, `publication`
- Core contracts: `WorkerDefinition`, `WorkerStatus`/health fields, `WorkerEvent`
- Required health fields exposed per worker:
  - `worker_id`, `worker_type`, `status`
  - `source_id`, `pipeline_id`, `modality`
  - `queue_depth`, `last_activity_at`, `last_success_at`
  - `error_count`, `processed_count`

### Processing queue semantics

Persistent lightweight queue/index files:

- `data/runtime/queues/pending_processing.jsonl`
- `data/runtime/queues/processing_claims.json`
- `data/runtime/queues/completed_processing.jsonl`
- `data/runtime/queues/failed_processing.jsonl`

No broker is introduced; workers coordinate through filesystem state.

### Claim semantics

- Workers claim a take before processing.
- Claim is released on completion/failure.
- Stale claims are recovered after timeout.
- Failed takes use bounded retry with backoff and `next_retry_at`.
- Duplicate processing is prevented through terminal/claim checks and worker markers.

### Acquisition -> processing orchestration

- Acquisition workers enqueue eligible takes with source/modality/purpose/fusion-readiness metadata (`acquisition_group_id`, `frameset_id`, capture timestamp).
- Processing workers resolve active bindings and execute immutable recipe-version runs using existing dispatch paths.
- Family-aware routing remains by modality/pipeline family; no cross-family silent fallback.

### Publication layer

- Publication workers consume completed processing entries and emit lightweight operator-facing `PublishedInspectionResult` summaries.
- Publication output remains independent from Studio debug artifact trees.

### Runtime execution environment hardening

- Runtime subprocess interpreter resolution is venv-safe and deterministic:
  - prefer `sys.executable`
  - fallback to `python3`
  - fallback to `python`
- FTP commands that start with `python -m ...` or `python3 -m ...` are normalized to the resolved interpreter.
- For `pyftpdlib` commands, `-d <upload_dir>` is injected when missing.
- If configured `-d` root differs from runtime upload dir, runtime emits `FTP_ROOT_MISMATCH` warning.

### Recovery semantics across short-lived CLI/API invocations

- Supervisor recovers active processes from persisted PID + instance metadata on each new process.
- Recovered running PID with matching ownership/signature is marked `running` without requiring in-memory continuity.
- Ownership mismatch warnings (`STALE_PID_OWNERSHIP_MISMATCH`) are emitted only after command/cwd/signature checks fail.

### Foreground debug mode

- CLI supports foreground runtime start:
  - `python scripts/runtime.py start trispector_ftp --foreground`
- This runs attached to the terminal for live debugging/log visibility while preserving the same normalized command preparation.

### Operator vs Studio separation

- Studio keeps stage-level engineering/debug views.
- Operator-facing publication is reduced to decision/result summaries and overlays.
- Runtime/worker APIs expose operational state without forcing raw pipeline complexity into operator flows.
- Processing Lab includes read-only active binding visibility (`source_id`, `modality`, `purpose`, `pipeline_id`, `active_recipe_version_id`).

Boundary rules:

- Runtime panel is engineering supervision.
- Operator UI remains the consumer of `PublishedInspectionResult` for production/operator decisions.
- Worker start/restart ownership remains CLI/supervisor-owned in this step; Studio only requests stop.
- No operator classification dashboard is added to Studio runtime views.

## TriSpector FTP Runtime Hardening (pyftpdlib)

- Runtime command normalization now enforces pyftpdlib write mode: if `-w` is missing, it is injected and event `FTP_WRITE_ENABLED` is emitted.
- FTP auth semantics are explicit in runtime config:
  - `auth_mode: anonymous` (default) emits `FTP_AUTH_ANONYMOUS`
  - `auth_mode: user_password` with `username/password` emits `FTP_AUTH_USER_PASSWORD`
- Runtime emits structured FTP diagnostics for SOPAS interoperability debugging:
  - `FTP_CLIENT_CONNECTED`
  - `FTP_CLIENT_LOGIN_SUCCESS`
  - `FTP_CLIENT_LOGIN_FAILED`
  - `FTP_UPLOAD_STARTED`
  - `FTP_UPLOAD_COMPLETED`
  - `FTP_UPLOAD_FAILED`
- FTP server stdout/stderr is forwarded into runtime logs via prefixed lines:
  - `[ftp_stdout] ...`
  - `[ftp_stderr] ...`

### FTP Selftest Workflow

- Local validation command:
  - `python scripts/runtime.py selftest trispector_ftp`
- The selftest attempts localhost FTP connection, auth, upload, on-disk verification, and cleanup.
- This is the canonical first check before SOPAS/hardware testing.

### FTP Status API

- `GET /api/runtime/processes/{process_id}/ftp-status`
- Returns runtime-oriented operational fields:
  - `listening`, `host`, `port`
  - `auth_mode`
  - `upload_dir`, `writable`
  - `active_clients`
  - `recent_uploads`
  - `last_upload_timestamp`

### TriSpector FTP Debugging Checklist

1. Confirm runtime process is `running` and FTP status reports `listening=true`.
2. Confirm configured port is reachable on the host and not blocked by firewall.
3. Confirm sensor host and runtime host are on the same subnet/routable path.
4. Run `python scripts/runtime.py selftest trispector_ftp` before SOPAS upload tests.
5. Verify `auth_mode` and credentials match SOPAS configuration.
6. Verify `upload_dir` path exists and is writable by runtime process owner.
7. Inspect runtime events for connection/login/upload/parse/processing boundaries.
8. If upload events exist but no take is created, focus on parsing errors (`FTP_UPLOAD_FAILED`/`PARSE_FAILED`).
# Engineering Debug Layer (Additive)

Sensor Studio now supports a renderer-driven `Operator` vs `Engineering` mode without route or pipeline forks.

- Operator mode keeps compact stage overlays and summaries.
- Engineering mode enables semantic diagnostics through existing stage registries and artifact contracts.
- This is additive: stages still publish native artifacts; renderers decide compact vs deep debug behavior.

## Engineering Renderer Model

- Mode toggle is Studio-level state, propagated to stage semantic renderers.
- Stage views remain stage-native; engineering-only tabs (`residuals`, `diagnostics`, `profiles`, `provenance`) are additional view ids.
- Inspectors consume stage artifacts/metadata, not ad-hoc side channels.

## Compatibility Direction

The model stays compatible with future `RGB + 25D` fusion, point-cloud viewers, multimodal overlays, ML classifiers, and live runtime monitoring because contracts remain artifact-first and stage-scoped.

## Studio vs Datasets UX composition boundary

This architecture now treats Studio and Datasets as complementary workspaces:

- Studio = engineering lab for acquisition-linked processing, stage diagnostics, overlays, compatibility checks, and reruns.
- Datasets = semantic curation and ML preparation layer for labeling, review/validation, object annotations, split management, and dataset composition.

UX composition rule:

- keep Studio left rail operational and compact
- move semantic curation/admin controls to Datasets
- preserve existing pipeline/run/stage contracts and processing backends

Refinement rule (lightweight UX pass):

- batch/admin actions in Studio are discoverable but visually de-emphasized
- orientation cards are compact and non-duplicative
- selector naming must distinguish acquisition navigation from semantic curation scopes

This is a composition refinement only. No processing contracts, run artifacts, or routing/back-end architecture are changed.

## Dataset Session Export Service

- Added compositional API service `DatasetSessionExportService` for deterministic session summary/export generation.
- New endpoints:
  - `GET /api/dataset-sessions/{session_id}/summary`
  - `GET /api/dataset-sessions/{session_id}/export`
- Service reuses existing take summary/detail + DatasetService metadata; it avoids coupling export DTOs to frontend-specific views.
- Export emphasizes metadata/references to maintain immutable acquisition and future import compatibility.

## ML Ingestion Wizard Backend Architecture

Added compositional backend modules and endpoints for ingestion runs.

### Core modules

- `ingestion_wizard.py`
- `label_manifest_builder.py`
- `take_reference_resolver.py`
- `range_expansion.py`
- `label_normalization.py`
- `object_grouping.py`
- `ml_set_materializer.py`

### API endpoints

- create/get ingestion run
- table ingestion
- reconciliation
- policy update
- canonical manifest generation
- deterministic materialization
- optional metadata application

This architecture preserves immutable acquisition semantics and keeps ingestion UX decoupled from runtime/training execution concerns.

## ML Set Summary / Governance Aggregation

- Added `ml_set_summary.py` as a compositional aggregation layer over existing ML-set memberships and DatasetService metadata.
- Summary/export endpoints provide lightweight governance views for the ML Set detail drawer:
  - summary
  - class distribution
  - split summary
  - warnings
  - representative samples
- deterministic export routing

## Physical Object Registry Architecture

- Added additive semantic registry and lightweight endpoints for object listing, detail, take linkage, repeatability summary, and reconciliation sync.
- Runtime pipelines remain decoupled; Physical Objects live in the semantic layer and are linked through sidecar metadata.
