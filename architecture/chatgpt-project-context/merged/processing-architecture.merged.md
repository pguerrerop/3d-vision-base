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

ChArUco detection diagnostics now include dictionary name, marker IDs, API mode, sharpness estimate, board coverage estimate, and failure reason to support fast debugging when marker-only detections occur.

2D calibration runtime now includes camera-control tuning plus live image diagnostics (histogram/clipping/brightness) to guide exposure/focus adjustments before corner detection.

Modal-based separation formalizes boundaries between runtime streaming/control state and persisted calibration capture/detection state.

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

## ProcessBinding + immutable recipe runtime (step 1)

Studio stays the editing/debugging surface, while runtime workers and manual execution resolve `ProcessBinding` when available.

- Resolve key: `source_id + modality + purpose`
- Bound target: `pipeline_id + active_recipe_version_id + optional calibration_profile_id`
- Purpose enum: `acquisition_inspection`, `manual_debug`, `fusion_input`, `fusion`

For process-service pipelines, execution is pinned to immutable recipe snapshots (`active_recipe_version_id`) instead of mutable instance state.

Manual Studio runs and worker runtime now share the same backend execution path and run metadata contract.

Persisted run metadata includes:

- `pipeline_id`
- `recipe_version_id`
- `config_snapshot_hash`
- `source_id`
- `take_id`
- `acquisition_group_id` (nullable)
- `calibration_profile_id` (nullable)

Fusion execution is intentionally deferred in this step.

## AcquisitionGroup for multimodal capture grouping (step 2)

System now supports `AcquisitionGroup` as the cross-modality synchronization boundary for the same conveyor event.

Fields:

- `id`
- `name` (nullable)
- `station_id` (nullable)
- `trigger_id` (nullable)
- `encoder_position` (nullable)
- `started_at`
- `completed_at` (nullable)
- `status` (`open | complete | failed`)
- `metadata`

Initial grouping workflow can be manual via API. Capture and processing paths propagate `acquisition_group_id` when present while preserving legacy behavior when omitted.

Future grouping can be trigger-based, encoder-position-based, or timestamp-window-based. Fusion is intentionally not implemented in this step.

## Acquisition split for 2D and 2.5D

- Acquisition remains a source/process boundary, separate from classification.
- 2D USB capture is a first-class acquisition source producing `rgb` takes.
- 2.5D TriSpector FTP ingest is a first-class acquisition source producing `heightmap` takes (labeled as `heightmap_2_5d` in take views).

### TriSpector FTP ingest responsibilities

- Ingest/upload stability and registration
- Parse using TriSpector parser contract
- Persist parsed assets + diagnostics + original upload
- Publish take as ready

FTP ingest explicitly does **not** run classification in the upload handler.

### TriSpector parser output contract

- `height16.tif` as `files.heightmap`
- `reflectance.png` as `files.reflectance` (when available)
- `parser_metadata.json` as parser diagnostics
- `heightmap_preview.png` as preview artifact
- original upload as `files.raw_upload`

### Binding connection after acquisition

- On take creation, resolve binding key:
  - `source_id`
  - `modality`
  - `purpose` (runtime defaults to `acquisition_inspection`)
- If active binding exists, trigger immutable recipe-version execution path.
- If no binding exists, keep ingestion successful and expose warning:
  - `No active processing binding found for this source/modality/purpose.`
- No modality-mismatched fallback for 2.5D acquisition paths.

Studio/manual and runtime acquisition-triggered processing remain aligned by using the same binding resolution semantics.

## Acquisition-to-processing seam (read-only contract)

Ownership boundaries:

- Acquisition owns ingestion/capture/parsing/upload/raw persistence/take creation.
- Processing owns binding resolution and execution through the shared runner path.

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

Modality normalization:

- Canonical processing modalities remain `rgb` (2D) and `heightmap` (2.5D).
- `heightmap_2_5d` is accepted as an alias and normalized to `heightmap`.
- Original value is preserved in `metadata.original_modality`.

Resolver service:

- `process_acquired_take_if_bound(...)` resolves active `ProcessBinding` by:
  - `source_id`
  - normalized `modality`
  - `purpose` (default `acquisition_inspection`)
- If `auto_process=False`: `ready_not_processed`.
- If no binding: `processing_binding_missing` warning, no failure.
- If binding exists: route through shared dispatch/runner path with bound `pipeline_id`, `recipe_version_id`, and optional `calibration_profile_id`.
- No hardcoded 2.5D fallback to 2D/3D pipelines.

Status lifecycle per take (`data/acquisition_processing/status/<take_id>.json`):

- `acquired`
- `ready_for_processing`
- `processing_binding_missing`
- `processing_enqueued`
- `processing_running`
- `processing_completed`
- `processing_failed`
- `ready_not_processed`

Run metadata persisted with seam context:

- `take_id`, `source_id`
- `acquisition_group_id`, `acquisition_process_id`, `acquisition_run_id`
- `pipeline_id`, `recipe_version_id`, `config_snapshot_hash`
- `modality`, `modality_family`

API seam endpoints:

- `POST /api/acquisition/processing/resolve`
- `GET /api/takes/{take_id}/acquisition-processing-status`

Studio visibility:

- `TakeSummary` and `TakeDetail` now carry `acquisition_processing_status` for generic status rendering in existing pages.

Fusion-forward note:

- `acquisition_group_id` remains the correlation key for future RGB+25D fusion execution.

## First RGB + 25D fusion preparation layer

Added a read-only group resolver that consumes existing processed outputs by `acquisition_group_id`.

### FusionInputBundle

- `acquisition_group_id`
- `rgb_take_id` / `heightmap_take_id`
- `rgb_run_id` / `heightmap_run_id`
- `rgb_result_payload` / `heightmap_result_payload`
- `rgb_artifacts` / `heightmap_artifacts`
- `readiness_status`
- `missing_inputs`
- `warnings`
- `created_at` / `resolved_at`

Readiness states:

- `waiting_for_rgb`
- `waiting_for_heightmap`
- `waiting_for_processing`
- `ready_for_fusion`
- `incomplete`
- `failed_input`

Selection behavior:

- Discover takes in group.
- Resolve RGB and 25D candidates.
- Choose latest successful candidate by default.
- Emit warnings and debug candidate lists when multiple candidates exist.

### Object pairing prep

Introduced `FusionObjectCandidate` for non-final debug alignment:

- `centroid_projection`
- `bbox_overlap`
- `object_index_fallback` (MVP fallback)
- `unmatched_rgb`
- `unmatched_heightmap`

Missing optional object fields do not fail resolution; warnings are returned.

### FusionPreviewResult and APIs

`FusionPreviewResult` includes:

- readiness
- pairing candidates
- merged feature summary
- recommended next action
- warnings

API endpoints:

- `GET /api/acquisition-groups/{group_id}/fusion-inputs`
- `GET /api/acquisition-groups/{group_id}/fusion-preview`

### Studio visibility

In Studio, when the selected take belongs to an acquisition group, the processing workspace now shows:

- RGB and 25D input readiness
- selected run IDs
- missing inputs
- pairing table and warnings

### Deferred items

- Calibration-aware geometric matching.
- Final fused operator classification/publication.

## Persisted RGB+25D fusion run path

Pipeline:

- `pipeline_id = mining_steel_ball_fusion_rgb_25d`
- `pipeline_family = fusion`
- `execution_backend = native`
- group-scoped by `acquisition_group_id`

Fusion run persistence:

- `data/fusion/runs/group_<acquisition_group_id>/run_<fusion_run_id>/`
- `fusion_run.json` stores run metadata:
  - pipeline id/family
  - source RGB/25D take + run refs
  - optional recipe/config snapshot refs
  - status/timestamps
  - result/artifact paths
  - warnings

Fusion result payload:

- `fusion_result.json`
- `fusion_summary.json`
- `fusion_object_table.json`
- `fusion_debug_pairing.json`

Final object payload includes:

- object linkage (`rgb_object_id`, `heightmap_object_id`)
- final class + class group
- confidence + explicit decision reasons
- RGB + 25D evidence blocks
- measurements + matching metadata

Rule-based MVP classification:

- 25D elongated/flat scrap-like -> `Chatarra`
- high 25D deformation -> `Bola deformada` (`Scrap de Bola`)
- RGB chip/crack hints -> `Bola con chip` / `Bola partida` (`Scrap de Bola`)
- healthy round consensus -> `Bola buena`

APIs:

- `POST /api/acquisition-groups/{group_id}/fusion-runs`
- `GET /api/acquisition-groups/{group_id}/fusion-runs`
- `GET /api/fusion-runs/{fusion_run_id}`
- `GET /api/acquisition-groups/{group_id}/fusion-result/latest`

Readiness enforcement:

- default: requires `ready_for_fusion`
- override: `force=true`

Operator publication is intentionally deferred; this step is for persisted fusion traceability/debug validation and reproducible review.

Future extension:

- bind fusion execution to process bindings/recipes (`purpose=fusion`) using existing metadata (`recipe_version_id`, `config_snapshot_hash`).

## ObjectCandidate semantic contract for pre-fusion pipelines (step 3)

System now standardizes inter-pipeline object semantics through `result_payload.object_candidates`.

- Contract is shared across 2D and 2.5D outputs.
- Legacy result fields remain backward compatible.
- Pipelines may still emit modality-specific artifacts and diagnostics.
- Empty detection cases emit `object_candidates: []`.

This semantic layer is the intended input to future fusion. Fusion logic remains intentionally deferred.

## Fusion pipeline over ObjectCandidates (step 4)

System now executes a first fusion pipeline (`mining_steel_ball_fusion`) over grouped 2D + 2.5D `ObjectCandidate` outputs.

- Fusion consumes per-modality `ObjectCandidate[]` by `acquisition_group_id`.
- Matching is initially pixel-space (centroid then bbox IoU fallback).
- Unmatched candidates remain explicit in `FusionResult`.
- `FinalObject` emits fused class/measurements with traceable decision reasons.

Calibrated world-space matching is deferred.

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

## Lightweight worker orchestration (local-first)

Runtime supervision remains process-oriented; workers now provide continuous local orchestration without distributed infrastructure.

### Worker model

- Worker classes/types: `acquisition`, `processing`, `fusion`, `publication`
- Worker state contracts include:
  - `worker_id`, `worker_type`, `status`
  - `source_id`, `pipeline_id`, `modality`
  - `queue_depth`, `last_activity_at`, `last_success_at`
  - `error_count`, `processed_count`

### Queue/index contract

- `data/runtime/queues/pending_processing.jsonl`
- `data/runtime/queues/processing_claims.json`
- `data/runtime/queues/completed_processing.jsonl`
- `data/runtime/queues/failed_processing.jsonl`

This queue is filesystem-backed and intentionally lightweight.

### Claim + retry semantics

- Worker must claim before processing.
- Claim is released on completion/failure.
- Stale claim recovery is timeout-based.
- Failures support bounded retries + backoff.
- Duplicate processing is prevented through claim/terminal checks and worker markers.

### Acquisition-processing flow

- Acquisition paths enqueue pending entries with fusion-readiness metadata:
  - `acquisition_group_id`
  - `frameset_id`
  - capture timestamp window fields
- Processing workers resolve bindings (`source_id + modality + purpose`) and execute immutable recipe versions using existing execution APIs.
- Family-aware routing is preserved.

### Publication layer

- Publication workers consume completed entries and emit operator-facing published summaries independent from Studio artifact trees.
- Operator/runtime remains summary-oriented; Studio remains artifact/stage-oriented.

### Runtime env + recovery hardening

- Runtime subprocess interpreter is resolved deterministically:
  - prefer `sys.executable`, then `python3`, then `python`.
- FTP commands using `python -m ...` / `python3 -m ...` are normalized automatically to the resolved interpreter.
- `pyftpdlib` commands auto-inject `-d <upload_dir>` when missing; mismatched roots emit `FTP_ROOT_MISMATCH`.

### Cross-invocation supervisor recovery

- New supervisor instances recover runtime state from PID + instance metadata and mark valid owned processes as `running`.
- Recovery does not depend on prior in-memory state (supports short-lived CLI/API invocations).
- Ownership mismatch warnings are guarded by command-signature and cwd checks to reduce false orphan detection.

### Foreground runtime CLI mode

- `scripts/runtime.py start <process_id> --foreground` runs runtime attached to terminal output for direct debugging.
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
