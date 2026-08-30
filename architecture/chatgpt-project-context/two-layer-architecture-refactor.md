# Two-layer architecture refactor (incremental)

## Goal

Refactor the POC into:

1. Reusable platform layer (acquisition/calibration/segmentation/debug/profiling)
2. Application layer (mining steel-ball inspection)

while preserving existing behavior and CLI compatibility.

## Implemented decisions

- Added `vision_3d_acquisition.vision_core` as the reusable platform namespace.
- Added source abstraction:
  - `AcquisitionSource` protocol
  - `FileSource` (filesystem queue implementation)
  - `ReplaySource` (filesystem-backed replay alias)
  - `SensorSource` (explicit placeholder for future live adapters)
- Extended capture references with available modalities, grouped assets, frame count, and raw metadata.
- Added minimal code-driven pipeline primitives:
  - `PipelineContext`
  - `PipelineStage`
  - `PipelineRunner`
  - `PipelineResult`
- Added modality metadata and validation:
  - take metadata supports `modalities` and `frameset`
  - supported modalities are `point_cloud`, `heightmap`, `reflectance`, `rgb`, `rgb_video`, and `laser_rgb`
  - stages can declare `required_modalities`
  - missing modality errors fail clearly before point-cloud stages run
- Added lightweight USB RGB acquisition validation:
  - OpenCV local camera discovery (`scripts/list_usb_cameras.py`)
  - image/video capture (`scripts/capture_usb_camera.py`)
  - optional synchronous API endpoints (`/api/cameras`, `/api/capture/image`, `/api/capture/video`)
  - RGB takes publish normal incoming folders, sessions, and runtime status without invoking point-cloud processing
- Added browser-first live preview:
  - acquisition overwrites throttled JPEG preview frames under `data/runtime/previews/`
  - API exposes `/api/runtime/preview` and `/api/runtime/preview/metadata`
  - Operations, Studio, Calibration, and Diagnostics poll preview metadata/image with stale/disconnected states
  - native OpenCV preview is now explicit engineering mode via `--preview-window`
- Added explicit source vs pipeline model:
  - acquisition sources produce raw/live data
  - processing pipelines consume modalities and expose registry metadata at `/api/pipelines`
  - result payloads can expose `stage_outputs`
  - Studio top-level route supports stage-by-stage inspection and future fusion placeholders
  - Studio evolved into a workstation layout with fixed shell, independently scrolling browser/workbench/inspector panes
  - stage selection drives center workbench content, tool availability, artifact focus, and inspector summaries
  - object candidates are selectable UI entities across segmentation/classification/measurement/future-fusion context
  - artifacts are typed, stage-provenanced engineering outputs routed through an artifact explorer
  - viewport contract is fixed-shell plus pane scrolling, not browser-page scrolling
  - calibration is positioned as source alignment/configuration with future slots for 2D camera, laser line, conveyor/world coordinates, encoder/sensor synchronization, and RGB-to-3D alignment
- Added reusable generic stages:
  - `LoadCaptureStage`
  - `DecodeHeightmapStage`
  - `ApplyCalibrationStage`
  - `PlaneFilterStage`
  - `SegmentObjectsStage`
  - `RunLegacySegmentationStage` (compatibility bridge)
- Added app split under `vision_3d_acquisition.apps`:
  - `acquisition_studio` app for reusable engineering/debug flow
  - `ball_inspection` app for domain stages
- Added ball-specific stages only in app layer:
  - `FitSphereOrEllipseStage`
  - `BallClassificationStage`
  - `StatisticsStage`
- Updated ball inspection default flow to be stage-native end-to-end (no `RunLegacySegmentationStage` in default path).
- Kept `RunLegacySegmentationStage` available for compatibility/testing only.
- Added new entry points:
  - `scripts/run_acquisition_studio.py`
  - `scripts/run_ball_inspection.py`
- Kept existing `scripts/process_latest_real.py` command untouched for backward compatibility.
- Added `--engine legacy|native` to `scripts/process_latest_real.py`; default remains `legacy`, while `native` routes through the stage-native ball inspection path.
- Added POC readiness support without changing architecture direction:
  - `vision_3d_acquisition.poc.summary` builds `poc_summary` and calibration diagnostics.
  - `vision_3d_acquisition.poc.labels` stores independent `data/takes/<take_id>/labels.json`.
  - `vision_3d_acquisition.poc.exports` provides labeled dataset and object-metrics exports.
  - `scripts/poc_tools.py` exposes summary, label, export, and validation helpers.
- Added lightweight acquisition session model (filesystem-backed at the time; sessions now live in the `data/index.db` catalog):
  - sessions stored under `data/sessions/<session_id>/`
  - `session_id` attached to takes and results
  - API supports session listing, summary, and take filtering
- Extended FrameSet contract for future synchronization:
  - `frameset_id`, `timestamp`, `assets`, and `synchronization.mode/confidence`
  - backward compatible with single-frame takes
- Added runtime acquisition status service and diagnostics:
  - `runtime.json` now includes connectivity, source details, preview freshness, queue, lag, stale, session, calibration, and warnings
  - SSE stream includes `runtime` events
- Added local single-process live pipeline loop:
  - `scripts/run_live_pipeline.py` watches incoming takes
  - auto-processes, updates runtime status, and supports graceful shutdown
- Added throughput diagnostics and warnings:
  - acquisition FPS vs processing FPS
  - latency, queue wait, debug/render overhead, export/write overhead
- Added calibration robustness helpers:
  - compatibility validation with confidence and age
  - recommended calibration selection by modality/sensor/setup recency

## Why no visual pipeline editor now

Stage contracts, runtime metrics needs, and operational workflows are still evolving. A visual editor is deferred to avoid locking into premature UX and plugin surfaces before the stage model is stable.

## Behavior preservation

- Existing real segmentation flow remains available and unchanged.
- `result.json` continues to validate against `ProcessingResult`.
- Ball inspection flow writes `result.json`, `DONE`, state files, and events in the same processed tree.
- Real result payloads include `processing_engine`, `calibration_diagnostics`, and `poc_summary` for operational validation and demo readiness.
- Real result payloads include `input_modalities`, `output_modalities`, `processing_pipeline`, and typed calibration details while preserving legacy calibration fields.
- Top-level UX is Operations / Studio / Calibration / Diagnostics. Operations separates machine status, raw acquisition sources, active processing pipeline, and latest inspection result. Studio is a three-region engineering workspace: persistent browser, central tabbed workbench, and contextual inspector. Diagnostics is the runtime health console.
- Calibration is currently typed as `plane_3d` with `source_modalities: ["point_cloud"]`; old calibration files default to this type.
- Canonical processing artifact model introduced: `result.artifacts` with typed stage/object-linked outputs and metadata.
- API take detail normalizes artifacts for both new explicit artifacts and legacy payload backfill.
- Overlay artifact model formalized: `kind: "overlay"` with `overlay_type`, `target_artifact_id`, `geometry`, `style`, and lineage fields.
- Overlay coordinate spaces formalized: `image_pixel`, `normalized_image`, `plot_pixel` (supported now), plus future `world_mm` and `point_cloud_projection`.
- Overlay target routing hardened: overlays render only when the declared target image artifact resolves; otherwise they remain metadata-only with warnings.
- Pipeline execution introspection added: `result.pipeline_execution` with per-stage status, timing, warnings/errors, and artifact/object counts.
- Studio execution graph added as a lightweight stage flow visualization bound to stage selection.
- Processing-unit registry metadata expanded with version/description/modality dependencies/artifact kinds/dependency graph/runtime support.
- Pipeline composition model added to registry (`execution_order`, `artifact_flow`, optional and conditional stages).
- Future 3D viewer contract prepared through `kind: "point_cloud"` artifact metadata (coordinate frame, units, point count, projection references).
- Canonical projection artifacts added (`xy_topdown`, `xz_side`, `yz_side`, `object_crop`) with deterministic coordinate metadata and transform ids.
- Overlay targeting now prefers projection artifacts with `projection_pixel` coordinates; legacy screenshot/plot overlays run in compatibility mode with explicit warnings.
