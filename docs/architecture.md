# Architecture

This document describes the high-level design of the 3D acquisition and processing POC (including SICK TriSpector/Ruler sensors).

## Two-layer architecture direction

The POC now uses a two-layer split:

1. `vision_core` (reusable platform layer)
   - Source abstractions (`SensorSource`, `ReplaySource`, `FileSource`)
   - Capture references with modalities, assets, metadata, and frame counts
   - Generic staged pipeline primitives (`PipelineContext`, `PipelineStage`, `PipelineRunner`)
   - Generic processing stages (capture load, cloud decode, calibration load, plane filtering, segmentation)
   - Shared serialization helpers
2. `apps/*` (application-specific layer)
   - `apps/acquisition_studio`: reusable engineering/debugging workflow
   - `apps/ball_inspection`: mining steel-ball classification workflow

This keeps acquisition/calibration/visualization/profiling reusable and avoids coupling generic infrastructure to one domain.

## Goals

- Capture 3D profile / point-cloud **takes** from a SICK sensor (or offline substitutes).
- Hand off takes to downstream processing without tight coupling.
- Keep the first implementation small, testable, and easy to extend (FTP, Harvesters/GigE, UI/API).

## Why independent processes

A single monolithic loop (acquire → process → display → actuate) is simple at first but becomes hard to evolve:

| Concern | Monolith | Separate processes |
|--------|----------|-------------------|
| Sensor I/O blocking | Blocks everything | Isolated in acquisition |
| Processing crashes | Can kill acquisition | Acquisition keeps running |
| Deploy / restart | All-or-nothing | Restart one role |
| Testing | Needs hardware mocks | Offline acquisition + real processing |
| Future UI/API | Embedded in same loop | Own service, own lifecycle |

For the POC we split **acquisition**, **processing**, and later **UI/API** and **output control** into separate programs that communicate only through agreed filesystem contracts. Each process owns its runtime, logging, and failure domain.

## Why a filesystem queue (POC)

For production you might use a message bus, object store, or database. For the POC we use directories under `data/`:

- **No extra infrastructure** — works on a laptop or edge box with only disk.
- **Human-debuggable** — inspect `metadata.json`, PLY, and marker files directly.
- **Atomic publish** — rename of a complete temp folder is a well-understood pattern on local disk.
- **Natural backpressure** — processing consumes folders when ready; incoming depth is visible.

Trade-offs (accepted for POC): no cross-host queue without shared storage, weaker ordering guarantees than a dedicated broker, and polling/watch overhead. These can be replaced later while keeping the same logical **take** contract.

## Repository layout

```
config/                      # acquisition.yaml, processing.yaml (from *.example)
samples/pointclouds/         # tracked test assets (e.g. sample.ply)
vendor/sick/sdk_4_3/         # SICK GenTL CTI (local SDK; check license before commit)
data/                        # runtime filesystem queue (gitignored)
docs/
scripts/                     # thin CLIs
vision_3d_acquisition/         # library code
  vision_core/               # reusable platform modules
  apps/                      # domain workflows
  poc/                       # run summaries, labels, diagnostics, exports
tests/
```

## Python package boundaries

| Package | Responsibility | Depends on |
|---------|----------------|------------|
| `contracts` | Stable Pydantic models / JSON schemas between processes | stdlib, pydantic |
| `vision_core` | Reusable acquisition/pipeline/serialization primitives | `contracts`, `processing`, `utils` |
| `apps` | App-specific orchestration and domain stages | `vision_core`, `contracts`, app logic |
| `poc` | POC readiness summaries, calibration health, labels, exports | `contracts`, stdlib |
| `acquisition` | Sources and modes: offline PLY, Harvesters, FTP | `contracts`, `storage`, `utils` |
| `storage` | Filesystem queue: stage, atomic publish, `READY` | `contracts`, `utils` |
| `processing` | Algorithms on takes (future) | `contracts`, `storage` |
| `state` | Acquisition/processing status files (future) | `contracts`, `utils` |
| `utils` | IDs, paths, timestamps | stdlib |

Rules:

- **contracts** — no I/O, no queue logic; only schemas and serialization.
- **vision_core** — generic components only; no ball domain imports or business terms.
- **apps** — owns domain-specific stages (classification decisions, app-level statistics).
- **poc** — owns operator-facing readiness summaries, dataset labels, validation helpers, and CSV/JSON exports; no database or orchestration framework.
- **acquisition** — produces takes; calls `storage.AcquisitionPublisher`, never writes `incoming/<id>/` without staging.
- **storage** — owns queue semantics; the only place that creates `.tmp`, renames, and touches `READY` for publishes.
- **processing** — consumes `incoming/` with `READY`, writes `processed/` (future).
- **state** — centralizes readers/writers for `data/state/*.json` beyond the minimal publish snapshot (future).

`acquisition.publisher` re-exports `AcquisitionPublisher` for backward compatibility; new code should import from `storage.publisher`.

## Process boundaries

```
┌─────────────────┐     data/incoming/<take_id>/      ┌──────────────────┐
│   Acquisition   │ ──────────────────────────────► │    Processing    │
│  (this repo v1) │         + READY marker            │    (future)      │
└─────────────────┘                                   └────────┬─────────┘
                                                                 │
                                                                 ▼
                                                        data/processed/
                                                                 │
                    ┌──────────────────┐                       │
                    │  UI / API        │ ◄── reads state, lists takes
                    │  (future)        │
                    └──────────────────┘
                                                                 │
                    ┌──────────────────┐                       │
                    │ Output controller│ ◄── optional actuation
                    │  (future)        │
                    └──────────────────┘
```

- **Acquisition** — sole writer to `data/incoming/` (via temp-then-rename). Writes `data/state/acquisition.json`.
- **Processing** — reads `incoming/` only when `READY` exists; writes results under `processed/` (future).
- **UI/API** — read-only on queue and state; may trigger commands via separate control channel later (not in v1).
- **Output controller** — consumes processed artifacts or commands (future).

No process should read partial takes: consumers wait for the `READY` marker.

## Stage-based pipeline model

`vision_core` introduces a minimal code-driven stage runner:

- `PipelineContext`: named artifacts + metrics + debug output registry.
- `PipelineStage`: small testable units with `run(context)`.
- `PipelineStage`: may declare `required_modalities`, `produced_modalities`, `produced_artifacts`, and `stage_category`.
- `PipelineRunner`: executes stages in order with profiling and validates declared modality requirements when the take reports modalities.
- `PipelineResult`: merged artifacts/metrics/profiling output.

Current stage split:

- Generic core stages:
  - `LoadCaptureStage`
  - `DecodeHeightmapStage`
  - `ApplyCalibrationStage`
  - `PlaneFilterStage`
  - `SegmentObjectsStage`
  - `RunLegacySegmentationStage` (compatibility bridge to existing implementation)
- Ball inspection domain stages:
  - `FitSphereOrEllipseStage` (POC placeholder)
  - `BallClassificationStage`
  - `StatisticsStage`

This is intentionally code-defined for now. A visual pipeline editor is explicitly deferred until stage contracts and operational needs stabilize.

## Multi-modal inspection model

The UI is organized as Operations / Studio / Calibration / Diagnostics. Operations is the production HMI. Studio is a three-region engineering workspace with a persistent data browser, central tabbed workbench, and contextual inspector. Calibration aligns/configures sources. Diagnostics investigates runtime health and payloads. Captures are no longer modeled as point-cloud-only.

The application shell uses fixed viewport layout (`100vh`, body overflow hidden) and delegates scrolling to the active panes. In Studio, the left data browser, center workbench, and right inspector are independent scroll regions. This follows industrial workstation patterns: navigation, work area, and context remain spatially stable while large artifacts, long JSON payloads, and inspector details stay accessible. Browser-page scrolling is intentionally disabled for the app shell; pane scrolling is the contract.

Studio rendering is stage-driven. Stage descriptors map to workspace tabs, expected artifacts, inspector summaries, and available object-focused tools. Selecting Segmentation prioritizes masks, clusters, candidate objects, and preprocessing diagnostics. Selecting Classification prioritizes object labels, confidence, rejection reasons, and fit/debug metadata. Selecting Measurements prioritizes dimensional tables, statistics, tolerances, and selected-object metrics. Fusion is represented as a future workspace for RGB/3D coexistence, synchronization, and calibration alignment without implying fusion algorithms exist today.

Studio now includes canonical spatial inspection and execution introspection:

- Spatial overlays (`kind: "overlay"`) are stage artifacts rendered as SVG over image previews.
- Object rows and overlay geometry are bidirectionally linked through `object_id`.
- Pipeline execution trace (`result.pipeline_execution`) exposes ordered stage diagnostics, including skipped/incompatible/failed stages.
- Lightweight execution graph (A -> B -> C) binds stage selection to workspace context.
- Overlay rendering is target-routed: overlays only render over their declared target image artifact, never over unrelated previews.
- Overlay coordinates are contract-driven (`image_pixel`, `normalized_image`, `plot_pixel`, future `world_mm`, `point_cloud_projection`) with approximate/non-renderable safeguards.

Projection architecture:

- canonical projection artifacts are generated for segmentation (`xy_topdown`, `xz_side`, `yz_side`)
- overlays target projection artifacts, not static matplotlib viewpoints
- compatibility mode preserves legacy screenshot overlays with explicit approximation warnings
- this separates engineering-view projections from debug-view screenshots and is the path toward calibrated spatial inspection UX

Object selection is held in UI state and applies across segmentation, classification, measurement, and future fusion views. The local identity is the result `object_id`; future tracking can layer temporal IDs on top. The artifact explorer treats generated outputs as typed engineering artifacts instead of passive image cards. Artifact descriptors carry stage provenance, type, status, optional file routing, and inspection metadata, leaving room for future split viewers, synchronized RGB/point-cloud panes, before/after comparisons, and replay.

Calibration is evolving from a single plane-calibration page into source alignment/configuration. The current implemented type remains `plane_3d`, but the UI and docs reserve space for 2D camera intrinsics/extrinsics, laser-line calibration, conveyor/world coordinate systems, ROI definition, encoder synchronization, sensor synchronization, and RGB-to-3D alignment. These are product and contract seams only; no new calibration algorithms are introduced here.

- **Take**: a capture folder plus metadata, assets, `READY`, and optional processed outputs.
- **Modality**: the input kind present in a take. Supported contract values are `point_cloud`, `heightmap`, `reflectance`, `rgb`, and `laser_rgb`.
- **Acquisition Source**: raw/live producer such as `usb_camera`, `offline_ply`, replay, or a future 3D sensor.
- **Pipeline**: processing definition that consumes modalities, such as `3d_ball_inspection` or future `2d_3d_fusion`.
- **Pipeline Stage**: ordered step such as segmentation, classification, measurement, or fusion.
- **Stage Artifact**: output produced by one stage, such as previews, masks, overlays, tables, statistics, point clouds, or JSON.
- **Object Candidate**: stable object identity within a take, used across segmentation, classification, measurements, and future tracking/fusion.
- **Capture asset**: a concrete file for one modality, for example `point_cloud.ply`, `point_cloud.npz`, `height.tiff`, `reflectance.png`, `rgb.png`, or `laser_overlay.png`.
- **FrameSet**: frame count plus optional synchronization and timestamp-source metadata for future multi-sensor captures.
- **Calibration type**: the current implementation is `plane_3d`, requiring `point_cloud`. Future UI slots exist for 2D camera, laser line, and fusion calibration without adding a large calibration framework now.
- **Pipeline modality requirements**: stages and app pipelines declare what they need. The current ball inspection path requires `point_cloud`; image-only takes fail before point-cloud stages run.

## Data directory layout

```
data/
  incoming/                    # acquisition → processing queue
    .<take_id>.tmp/            # in-progress (hidden); not consumed
    <take_id>/                 # published take
      metadata.json
      point_cloud.ply          # optional
      height.tiff              # optional
      reflectance.png          # optional
      READY                    # empty marker: safe to consume
  processed/                   # processing output (future)
    <take_id>/
      ...
      DONE                     # optional: processing complete (future)
  state/
    acquisition.json         # last publish / acquisition health (v1)
    runtime.json             # live runtime status + throughput diagnostics
  sessions/
    <session_id>/
      metadata.json          # acquisition session metadata
      takes/
        <take_id>/metadata.json
```

## Acquisition sessions

Sessions are lightweight filesystem groupings of related takes. They are used for operator trust and repeatable debugging, not as a database replacement.

- Session metadata includes operator, setup, mode, calibration, conveyor speed, encoder flag, notes, and creation timestamp.
- Live publishing can auto-create a session id when one is not provided.
- Takes remain valid standalone; `session_id` is additive metadata.

## Runtime acquisition state

`data/state/runtime.json` now carries machine-state diagnostics for live operation:

- connectivity (`acquisition_connected`, `acquisition_source`)
- queue health (`queue_size`, `dropped_frames`, `processing_lag_ms`)
- freshness (`latest_frame_timestamp`, `stale`)
- context (`current_session`, `active_calibration`)
- throughput (`acquisition_fps`, `processing_fps`, latency and overhead rolling metrics)

## Take lifecycle

1. **Allocate** `take_id` (timestamp-based; see `vision_3d_acquisition.utils.ids`).
2. **Stage** under `data/incoming/.<take_id>.tmp/` — copy assets, write `metadata.json`.
3. **Publish** — atomic rename to `data/incoming/<take_id>/`.
4. **Signal** — create empty `READY` file.
5. **State** — update `data/state/acquisition.json`.
6. **Consume** (future) — processing watches `incoming/`, processes folders with `READY`, moves or marks `processed/`.

## Marker files

| File | Location | Meaning |
|------|----------|---------|
| `READY` | `incoming/<take_id>/` | Acquisition finished; folder is complete |
| `DONE` | `processed/<take_id>/` (future) | Processing finished |

Rules:

- Downstream **must not** read `incoming/<take_id>/` until `READY` exists.
- `READY` is created **after** the folder rename (publish is atomic; marker is separate so consumers never see a half-renamed tree).

## Atomic publish convention

1. Write everything under `data/incoming/.<take_id>.tmp/`.
2. When complete, `rename(.<take_id>.tmp → <take_id>)` on the same filesystem.
3. Create `READY` in the final folder.

If acquisition crashes mid-stage, only `.tmp` remains; operators can delete stale `.tmp` directories. No `READY` means no consumer action.

## Implementation map (v1)

| Component | Package path | Status |
|-----------|--------------|--------|
| Metadata contracts | `vision_3d_acquisition.contracts` | Done |
| Publisher | `vision_3d_acquisition.storage.publisher` | Done |
| Offline PLY acquisition | `vision_3d_acquisition.acquisition.offline_ply` | Done |
| Live sensor (Harvesters) | `vision_3d_acquisition.acquisition.*` | Planned |
| FTP ingest | `vision_3d_acquisition.acquisition.*` | Planned |
| Processing worker | separate process | Planned |

See also: [processes.md](processes.md), [contracts.md](contracts.md), [acquisition.md](acquisition.md).

## Studio artifact bridge

`result.artifacts` is the canonical stage-output bridge. API take detail normalizes artifacts for both explicit (new runs) and derived (legacy result payloads) modes so frontend consumers do not branch on old vs new formats.

## Pipeline composition model

Processing pipelines are now documented as compositions of reusable processing units:

- `dependencies`: upstream stage requirements
- modality dependencies (`required_modalities`, `optional_modalities`)
- artifact flow (`composition.artifact_flow`)
- canonical order (`composition.execution_order`)
- optional/conditional stages (`composition.optional_stages`, `composition.conditional_stages`)

This keeps runtime behavior code-driven while stabilizing composition contracts for future expansion without introducing orchestration infrastructure.

## Future 3D viewer contract

No 3D renderer is implemented yet. Contracts are prepared via `kind: "point_cloud"` artifacts with metadata for:

- coordinate frame
- units
- point count and bounds
- projection references for 2D overlay compatibility

Future viewers should consume point-cloud artifacts and optionally project object overlays using shared artifact references (`target_artifact_id`, `projection_references`).
