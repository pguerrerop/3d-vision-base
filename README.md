# 3D Acquisition POC

Filesystem-queue based 3D acquisition and processing proof-of-concept for profile and point-cloud data (including SICK TriSpector/Ruler sensors).

## Documentation

- [Architecture](docs/architecture.md) — process boundaries, package layout, queue design
- [Processes](docs/processes.md) — acquisition, processing, UI/API roles
- [Contracts](docs/contracts.md) — JSON and filesystem contracts
- [Acquisition](docs/acquisition.md) — v1 acquisition implementation
- [Processing](docs/processing.md) — real segmentation pipeline and limitations
- [Calibration](docs/calibration.md) — web-based plane calibration and semantic labeling

## Quickstart

### 1. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install pytest   # for tests
```

### 3. Publish the tracked sample PLY as a take

```bash
python scripts/publish_ply_take.py \
  --ply samples/pointclouds/sample.ply \
  --data-dir data
```

Example output:

```
take_id: 2026-05-16T153012_001
modalities: point_cloud
folder: /path/to/repo/data/incoming/2026-05-16T153012_001
```

### 4. Expected layout

```
data/
  incoming/
    <take_id>/
      metadata.json
      point_cloud.ply
      point_cloud.npz
      READY
  processed/
    <take_id>/
      result.json
      input_point_cloud_preview.png
      overlay.png
      debug_height.png
      debug_segmentation.png
      DONE
  state/
    acquisition.json
    latest.json
    runtime.json
  events/
    events.jsonl
  sessions/
    <session_id>/
      metadata.json
      takes/
        <take_id>/
          metadata.json
```

Downstream processes should only consume folders that contain `READY`.

### 5. Mock-process the latest take

```bash
python scripts/mock_process_latest.py --data-dir data
```

This writes a `result.json`, `input_point_cloud_preview.png`, placeholder processed-output images, `DONE`, state, and event files for UI and contract testing alongside the real segmentation pipelines.

Mock processing is intentionally explicit:

- `processing_mode` is `"mock"` in `result.json`.
- The point-cloud preview and `input_stats` are computed from the real original `point_cloud.ply`.
- Object detections, decisions, overlay, height, and segmentation images are synthetic demonstration data.
- The operator and debug UI show a visible **MOCK PROCESSING** label so demo output is not mistaken for real industrial classification.

### 6. Real segmentation-process the latest take (legacy compatibility)

```bash
python scripts/process_latest_real.py --data-dir data
```

This runs the first real geometric pipeline: point-cloud loading, preprocessing, dominant plane removal, foreground clustering, real measurements, and real debug images. It writes `processing_mode: "real"` and `algorithm_stage: "segmentation"`. Object classes remain `"unknown"` and decisions remain `REVIEW` because industrial classification is not implemented yet. Processing CLIs print available modalities, required modalities, calibration type/source, and abort clearly when a take lacks `point_cloud`.

`process_latest_real.py` now also accepts `--engine legacy|native`. The default is still `legacy`, preserving existing command behavior. Use `--engine native` to run the future stage-native ball inspection pipeline while keeping the same processed folder contract (`result.json`, `DONE`, state, events, debug artifacts, and timing/profiling fields).

Every real processing run now includes a `poc_summary` in `result.json` and prints a compact CLI summary with demo readiness, object counts, timing, and warnings.

### 6b. Run acquisition studio debug flow (new reusable app layer)

```bash
python scripts/run_acquisition_studio.py --data-dir data --profile
```

This is the reusable engineering/debugging flow with no ball-specific assumptions. It loads the latest ready capture (or `--take-id`), decodes runtime cloud input, applies optional calibration, runs plane filtering + clustering, and prints per-stage timing.

Useful options:

- `--take-id <capture_id>` to inspect a specific capture.
- `--calibration config/calibrations/<file>.json` to apply a calibration.
- `--no-preview` to skip preview rendering.
- `--no-prefer-fast-cloud` to force `point_cloud.ply` instead of `point_cloud.npz`.

### 6c. Run ball inspection flow (new domain app layer)

```bash
python scripts/run_ball_inspection.py --data-dir data --profile
```

This runs the domain pipeline for mining steel-ball inspection. It consumes a capture source, executes segmentation plus domain stages (fit/classification/statistics), and writes `data/processed/<take_id>/result.json` plus `DONE`, state, and event updates.

The default ball inspection flow is stage-native end to end and does not use the legacy bridge stage.

Useful options:

- `--take-id <capture_id>` to process a specific capture.
- `--config config/processing.yaml` for stage thresholds.
- `--calibration config/calibrations/<file>.json` to apply calibrated filtering.
- `--skip-debug-images` for production latency profiling.
- `--no-prefer-fast-cloud` to force `point_cloud.ply` loading.

### 6d. POC review, labels, and exports

```bash
python scripts/poc_tools.py --data-dir data summary <take_id>
python scripts/poc_tools.py --data-dir data label <take_id> --label ball --label uncertain --reviewer operator
python scripts/poc_tools.py --data-dir data export-labels --output data/labeled_summary.csv
python scripts/poc_tools.py --data-dir data export-objects --output data/object_metrics.csv
python scripts/poc_tools.py --data-dir data validate-result <take_id>
```

Labels are stored independently from processing output at `data/takes/<take_id>/labels.json`, so they survive reprocessing and are ready for future ML dataset tooling. Object exports include diameter, fit error, point count, bounding box, confidence, and rejection fields.

### 6e. Run live processing loop

```bash
python scripts/run_live_pipeline.py \
  --data-dir data \
  --engine legacy \
  --poll-interval 0.5 \
  --session session_demo_01
```

The loop watches new `incoming/*/READY` takes, processes automatically, updates runtime acquisition status, tracks throughput warnings, and links takes into an acquisition session.

### 6f. Validate USB RGB acquisition

USB camera support is a lightweight local acquisition adapter for validating the multi-modal platform. It uses OpenCV `VideoCapture`; it does not run object detection and does not feed RGB takes into the point-cloud ball pipeline.

```bash
python scripts/list_usb_cameras.py --max-index 8
python scripts/capture_usb_camera.py --camera-index 0 --mode image --data-dir data --session session_usb_01
python scripts/capture_usb_camera.py --camera-index 0 --mode video --duration 10 --data-dir data --session session_usb_01
```

The browser is the primary preview surface. Active USB acquisition overwrites `data/runtime/previews/usb_camera_0.jpg` at a throttled rate (default 250 ms) and writes sidecar metadata for freshness/FPS diagnostics. The API exposes it at `GET /api/runtime/preview` and `GET /api/runtime/preview/metadata`, and Operations, Studio, Calibration, and Diagnostics poll those endpoints where relevant.

Add `--preview-window` only for the optional OpenCV engineering window. In image mode, SPACE captures and `q` quits. Captures publish normal `data/incoming/<take_id>/` folders with `modalities: ["rgb"]` for images or `["rgb", "rgb_video"]` for videos, update `runtime.json`, and attach to the selected session.

### 7. Start the backend

Copy `.env.example` to `.env` and adjust `API_PORT` if needed.

```bash
python scripts/run_api.py
```

Equivalent:

```bash
python -m vision_3d_acquisition.api.serve
```

Useful check:

```bash
curl http://localhost:${API_PORT:-8000}/api/health
```

### 8. Start the frontend

Copy `frontend/.env.example` to `frontend/.env` and adjust `VITE_PORT` and `VITE_API_PORT` if needed.

```bash
cd frontend
npm install
npm run dev
```

Open:

- `http://localhost:5173/operations` for the DevAI-branded production HMI
- `http://localhost:5173/studio` for sessions, takes, pipeline selection, stage exploration, and artifacts
- `http://localhost:5173/calibration` for plane calibration, inferred footprints, and semantic labeling
- `http://localhost:5173/diagnostics` for runtime health, profiling, payloads, and queue/FPS diagnostics

The frontend uses `VITE_API_BASE_URL` when set, otherwise it calls same-origin `/api` and the Vite dev server proxies to `VITE_API_HOST:VITE_API_PORT`.

The UI header uses a compact DevAI wordmark with the subtitle “Industrial Vision POC” across Operations, Studio, Calibration, and Diagnostics. Operations separates machine status, raw acquisition sources, active pipeline health, and latest processed inspection result. Studio is the engineering vision workspace: persistent data browser on the left, tabbed modality/stage workbench in the center, and contextual inspector on the right. The application shell uses a fixed `100vh` viewport with browser-page scrolling disabled; Studio delegates scrolling to the browser, workbench, and inspector panes independently so large artifacts, JSON payloads, and long contextual sections stay reachable without moving the whole application shell. Calibration is source alignment/configuration, not classification review. Diagnostics is the runtime console for health, profiling, payloads, warnings, and performance.

## Multi-modal inspection model

A **take** is one capture event with `metadata.json`, one or more capture assets, and a `READY` marker. Each take now reports `modalities`, such as `point_cloud`, `heightmap`, `reflectance`, `rgb`, `rgb_video`, or `laser_rgb`. A **capture asset** is a file referenced by `metadata.files` and grouped by modality in the API. A **FrameSet** records frame count and synchronization metadata (`none`, `timestamp`, `hardware_trigger`) for future multi-sensor captures. A **session** groups related takes for one acquisition run without adding a database.

An **acquisition source** produces raw/live data (`usb_camera`, `offline_ply`, future `trispector`/`ruler3000`). A **processing pipeline** consumes modalities and produces processed outputs. A **stage** is one named pipeline step, and a **stage artifact** is an image, point cloud, overlay, table, statistic, or JSON output from that stage. Current processing remains point-cloud based, but the contracts no longer assume every take is only a point cloud. Pipeline stages can declare `required_modalities`; the 3D ball inspection pipeline requires `point_cloud` and fails with a clear modality error for image-only takes. Fusion pipelines are represented as future metadata only until real RGB+3D synchronization and projection exist.

Studio follows the workflow: **Sessions → Takes → Modalities → Pipelines → Stages → Artifacts → Results**. A take is a multimodal container; pipelines consume subsets of those modalities. RGB-only takes can be inspected as inputs, but 3D Ball Inspection shows an incompatible state until a `point_cloud` modality is present.

Studio is stage-centric and object-centric. Selecting a pipeline stage changes the workbench, available tools, active tab, artifact focus, and inspector. Candidate objects are first-class UI entities (`Object #1`, `Object #2`, ...); selecting an object updates classification and measurement context, preparing the surface for future tracking and multi-camera fusion. Artifacts are explicit engineering outputs with type, producing stage, status, timestamp/provenance copy, and preview routing. This follows the common industrial vision split between acquisition inputs, task/stage outputs, results, and interfaces, while staying compatible with future Ruler3000/Ranger3 and GenIStream-style source/artifact workflows.

Studio now includes:

- canonical spatial overlays (`kind: "overlay"`) rendered over image artifacts
- bidirectional object/overlay hover and selection linking
- pipeline execution trace diagnostics (`result.pipeline_execution`)
- lightweight execution graph for stage-to-stage introspection

Projection and overlay architecture:

- canonical projection artifacts (`xy_topdown`, `xz_side`, `yz_side`, object crops) are emitted as normal artifacts
- overlays target projection artifacts via `target_artifact_id` using stable `projection_pixel` coordinates
- legacy matplotlib/debug screenshot overlays remain supported in compatibility mode with explicit approximation warnings

## Current processing limitations

The POC has real geometric segmentation and calibration-aware foreground filtering, but no final industrial object classification yet. `scripts/mock_process_latest.py` remains available to prove the filesystem contract and UI flow with demo data. Real processing writes `processing_mode: "real"` and geometric clusters with unknown class until classification is implemented.

### 9. Run tests

```bash
pytest tests/ -q
```

## Project layout

```
config/                  # Example YAML/JSON (copy *.example → local files)
samples/pointclouds/     # Tracked test PLY files
vendor/sick/sdk_4_3/     # SICK GenTL CTI (local install; gitignored)
vision_3d_acquisition/     # Python package
  vision_core/           # Reusable acquisition + processing platform
  apps/                  # acquisition_studio, ball_inspection
  contracts/             # Pydantic schemas (process boundaries)
  acquisition/           # Offline PLY publish (Harvesters/FTP planned)
  calibration/           # Plane detection and belt calibration storage
  storage/               # Filesystem queue publish/consume
  api/                   # FastAPI REST + SSE event stream
  processing/            # Geometric segmentation pipeline
  state/                 # Runtime status helpers
  utils/
frontend/                # React/Vite Operations, Studio, Calibration, Diagnostics UI
scripts/                 # Thin CLI entry points
docs/
tests/
data/                    # Runtime queue (gitignored)
```

Copy example config when wiring acquisition and runtime defaults:

```bash
cp config/acquisition.yaml.example config/acquisition.yaml
cp config/runtime.json.example config/runtime.json
```

Set the active default calibration from the Calibration UI or `config/runtime.json` (`default_calibration_file`).

## Command reference

- Legacy real segmentation flow (kept working):
  - `python scripts/process_latest_real.py --data-dir data`
- New acquisition studio debug flow:
  - `python scripts/run_acquisition_studio.py --data-dir data --profile`
- New ball inspection flow:
  - `python scripts/run_ball_inspection.py --data-dir data --profile`

## Status

- **Done:** offline PLY acquisition, filesystem publisher (`storage`), Pydantic contracts, real geometric processing, web calibration, FastAPI API with SSE, React Operations/Studio/Calibration/Diagnostics UI, mock processing for demos, reusable `vision_core` stage pipeline, acquisition studio and ball inspection app split
- **Planned:** Harvesters live grab and GenTL/CTI streaming path, FTP ingest, industrial classification, production decision rules, PLC/output integration

## Processing units and artifacts

Studio now uses a canonical processing artifact contract (`result.artifacts`) as the bridge between backend stages and UI stage tabs/explorer/inspector. New runs emit explicit artifacts; old `result.json` payloads are backfilled by API normalization for compatibility. See [docs/processing-units.md](docs/processing-units.md).

Processing units are now formally registered with metadata (`stage_id`, version, modality requirements, produced artifact kinds, dependencies, realtime support, optional/conditional flags). Pipeline definitions include a composition block describing execution order and artifact flow.

Point-cloud artifacts (`kind: "point_cloud"`) include forward-compatible metadata for future 3D viewers (coordinate frame, units, point counts, projection references). 3D rendering is still intentionally deferred.
