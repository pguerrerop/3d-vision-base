# Processing

Processing is now split into:

- reusable platform flow (`scripts/run_acquisition_studio.py`)
- ball inspection application flow (`scripts/run_ball_inspection.py`)
- legacy compatibility flow (`scripts/process_latest_real.py`)
- live runtime loop (`scripts/run_live_pipeline.py`)

The geometric segmentation path remains the core implementation and is reused by the new ball inspection pipeline.

## Source vs Pipeline Model

Acquisition sources and processing pipelines are separate concepts:

- Sources produce raw/live data and preview state: `usb_camera`, `offline_ply`, replay, future 3D sensors.
- Pipelines consume modalities and produce processed outputs: `3d_ball_inspection`, future `rgb_segmentation`, future `rgb_ball_classifier`, future `2d_3d_fusion`.
- Stages are ordered pipeline steps such as segmentation, classification, measurement/statistics, and fusion.
- Stage artifacts are the outputs of a specific stage: previews, masks, overlays, point clouds, tables, statistics, or JSON.

The pipeline registry is intentionally lightweight and exposed at `GET /api/pipelines`. It is metadata, not a plugin system or orchestration engine.

## Pipeline

Legacy command:

`scripts/process_latest_real.py --data-dir data`

This finds the newest `data/incoming/<take_id>/` folder with `READY` and no processed `DONE`, then writes:

```text
data/processed/<take_id>/
  result.json
  input_point_cloud_preview.png
  debug_plane_segmentation.png
  debug_foreground.png
  debug_clusters.png
  DONE
```

The default engine is `legacy`, so existing command behavior is unchanged. The same command also supports:

```bash
python scripts/process_latest_real.py --data-dir data --engine native
```

`native` routes the legacy CLI entrypoint through the stage-native ball inspection flow and is the future path for production processing. It preserves the processed output contract (`result.json`, state, events, debug artifacts, and `timing_ms`/`profiling`) while adding the ball inspection classification stages. `legacy` remains available as the compatibility path.

The legacy implementation lives in `vision_3d_acquisition.processing.pipeline`; native processing uses `vision_core` stages shared by app-level pipelines. Both engines use the shared processing-output writer for the public filesystem contract.

Both engines add `processing_engine`, `calibration_diagnostics`, and `poc_summary` to `result.json`. The POC summary is the operator-facing health check for demo readiness, calibration validity, warnings, object counts, artifacts, and bottlenecks.

Calibration mode is explicit:

- `plane_mode: "calibrated"` means a saved calibration file was used.
- `plane_mode: "auto"` means automatic plane estimation only.
- `plane_mode: "disabled"` is reserved for future explicit no-plane workflows.

When no calibration is provided, manual CLIs print a warning because auto plane estimation can produce plausible but less repeatable results. Use `--yes` or `--non-interactive` for automation, and `--require-calibration` for demo/customer-facing runs.

New ball inspection entrypoint:

```bash
python scripts/run_ball_inspection.py --data-dir data --profile
```

This is the production-style native entrypoint. If `--take-id` is omitted it selects takes in this order:

1. newest ready incoming take without `data/processed/<take_id>/DONE`
2. newest ready incoming take, even if already processed

An explicit `--take-id` overrides queue selection. Before processing starts, the CLI prints the selected take, available modalities, required modalities, selected calibration type/source, engine, and `Processing...`. If the take only has `rgb`, for example, the ball pipeline aborts with `Pipeline requires point_cloud, but take has rgb.`

The default ball inspection flow is stage-native end-to-end and no longer uses the legacy bridge stage. It executes explicit core stages (`LoadCaptureStage`, `DecodeHeightmapStage`, `ApplyCalibrationStage`, `PreprocessPointCloudStage`, `PlaneFilterStage`, `SegmentObjectsStage`, `MeasureObjectsStage`) followed by domain stages (`FitSphereOrEllipseStage`, `BallClassificationStage`, `StatisticsStage`) and result serialization.

## Modality Requirements

The foundation supports multiple capture modalities without changing the current point-cloud implementation:

- `metadata.modalities` lists available inputs: `point_cloud`, `heightmap`, `reflectance`, `rgb`, `rgb_video`, and `laser_rgb`.
- Source references expose take id, metadata, grouped assets, available modalities, and frame count.
- Stages can declare `required_modalities`. Point-cloud stages declare `point_cloud`.
- `PipelineRunner` validates requirements before each stage when modalities are known and records the missing-modality error in profiling metadata.
- `result.json` records `input_modalities`, `output_modalities`, and `processing_pipeline.required_modalities`.

Current ball inspection requires `point_cloud`. RGB, RGB video, reflectance, laser, and fusion processing are intentionally not implemented yet.

## USB RGB Acquisition Validation

The USB camera adapter is an acquisition milestone, not a computer-vision pipeline. It validates local camera discovery, image/video capture, modality-aware take publishing, sessions, runtime state, and browser UX.

```bash
python scripts/list_usb_cameras.py --max-index 8
python scripts/capture_usb_camera.py --camera-index 0 --mode image --data-dir data
python scripts/capture_usb_camera.py --camera-index 0 --mode video --duration 10 --data-dir data
```

Image takes publish `rgb.png` with `modalities: ["rgb"]`. Video takes publish `rgb_video.mp4` when an MP4 codec is available, fall back to an AVI file when needed, and include `preview.png`; their modalities are `["rgb", "rgb_video"]`. Metadata records camera index, backend, resolution, FPS, duration, frame count, and `timestamp_source: "usb_camera"`.

Live preview is browser-first and polling-based. During active USB acquisition, the runtime overwrites a single JPEG preview frame under `data/runtime/previews/usb_camera_0.jpg` at a throttled interval (default 250 ms, around 4 FPS) and writes `usb_camera_0.json` metadata with timestamp, resolution, stale flag, and FPS estimate. This avoids websocket video transport, HLS/WebRTC, media servers, and unbounded preview history.

The optional API endpoints are synchronous and local:

- `GET /api/cameras`
- `POST /api/capture/image`
- `POST /api/capture/video`
- `GET /api/runtime/preview`
- `GET /api/runtime/preview/metadata`

RGB-only takes should appear in Debug and sessions, but point-cloud stages must reject them clearly. That refusal is expected behavior until a real RGB processing pipeline is added.

Use `--preview-window` only when an engineer wants the native OpenCV window and keyboard shortcuts. Operational monitoring should use the browser live preview.

Studio uses the same registry to show compatibility. The future fusion pipeline declares `required_modalities: ["point_cloud", "rgb"]`, but remains disabled with “Fusion pipeline not implemented yet.”

Studio is organized around the engineering chain:

```text
Sessions -> Takes -> Modalities -> Pipelines -> Stages -> Artifacts -> Results
```

A take is a multimodal container. The Inputs tab shows only modalities available in the selected take. Stage selection drives the active workspace tab, visible artifacts, object list, available actions, and inspector context. Diagnostics-only runtime internals stay out of Studio except for concise compatibility, warning, and timing signals.

Stage-centric behavior:

- Segmentation prioritizes masks, clusters, candidate objects, segmentation artifacts, warnings, and timing.
- Classification prioritizes object labels, confidence, rejection reasons, fit metrics, and object-centric inspection.
- Measurements prioritizes diameter/statistics tables and selected-object measurements.
- Fusion is a placeholder workspace for future RGB/3D synchronization, calibration alignment previews, and multimodal processing.

Artifacts are routed through a lightweight explorer with type, stage provenance, status, file reference, and preview support. This mirrors industrial vision task/result separation without adding a node editor or execution engine.

The workspace uses an engineering-shell scroll model: the browser page is fixed to the viewport, and Studio’s data browser, central workbench, and inspector scroll independently. This keeps the source browser, selected stage, and context panel spatially stable while still allowing large overlays, point-cloud previews, tables, and JSON payloads to be inspected.

Object candidates are first-class in Studio. The current stable identity is `object_id` within a take; selecting an object carries across segmentation, classification, measurements, and the future fusion placeholder. That prepares the UI for later object tracking, multi-camera correlation, and replay without changing today’s processing algorithms.

New acquisition/debug entrypoint:

```bash
python scripts/run_acquisition_studio.py --data-dir data --profile
```

Live loop entrypoint:

```bash
python scripts/run_live_pipeline.py --data-dir data --engine legacy --poll-interval 0.5
```

The live loop is intentionally simple and local-first:

- polls for new ready takes
- processes in a single process
- updates `data/state/runtime.json` on each iteration
- computes rolling throughput warnings (processing slower than acquisition, queue buildup, debug/export overhead)
- gracefully shuts down on SIGINT/SIGTERM

## Runtime Point Cloud Format

PLY is kept as the debug/archive/download format. It is human-portable, but loading a large ASCII or general PLY through Open3D can dominate production latency.

`point_cloud.npz` is the runtime format. It stores:

- `points`: `float32` array with shape `Nx3`
- `colors`: optional `float32` array with shape `Nx3`
- `normals`: optional `float32` array with shape `Nx3`

Offline PLY publishing writes both files and metadata includes both references:

```json
{
  "files": {
    "point_cloud": "point_cloud.ply",
    "point_cloud_npz": "point_cloud.npz"
  }
}
```

Real processing prefers `point_cloud.npz` when it exists, so the profile shows `load_point_cloud_fast`. Use `--no-prefer-fast-cloud` to force the PLY fallback for compatibility testing. The live sensor path should bypass PLY entirely and publish or pass runtime point arrays directly.

## Generic Mode

Without a calibration file, processing uses generic dominant-plane segmentation:

1. Load the source `point_cloud.ply` with Open3D.
2. Voxel-downsample and remove statistical outliers.
3. Fit the dominant plane with Open3D RANSAC.
4. Treat plane inliers as background and outliers as foreground.
5. Cluster foreground points with DBSCAN.
6. Measure each cluster: point count, centroid, axis-aligned bounds, dimensions, and approximate diameter.
7. Render debug images from actual geometry.
8. Write `result.json` with `processing_mode: "real"` and `algorithm_stage: "segmentation"`.

## Calibrated Mode

Run calibrated processing with:

```bash
python scripts/process_latest_real.py \
  --data-dir data \
  --calibration config/calibrations/belt_setup_2026_05_16.json
```

The explicit `--calibration` argument wins. If it is omitted, the script uses `default_calibration_file` from `config/runtime.json` (set via the Calibration UI **Set as active/default**, the API, or CLI `--set-default-calibration`). Copy `config/runtime.json.example` to `config/runtime.json` for a starting template.

```json
{
  "default_calibration_file": "config/calibrations/belt_setup_2026_05_16.json"
}
```

If `config/processing.local.json` still contains `default_calibration`, that value is ignored and a deprecation warning is printed.

When a calibration is present, processing uses the saved semantic labels:

1. Use the plane labeled `belt` as the reference plane.
2. Crop early to the belt polygon or belt bounding box plus `calibration_crop_margin_mm`.
3. Remove points close to planes labeled `outer_plane_ignore`.
4. Compute height above the belt plane in millimeters.
5. Keep candidate foreground points only when they are inside `roi_polygon_xy_mm` and within the configured height range.
6. Cluster candidate foreground points with DBSCAN.
7. Keep clusters only when their center and configured fraction of points are inside the belt polygon.
8. Write `result.json` with `algorithm_stage: "calibrated_segmentation"`, `calibration_id`, `calibration_file`, `calibration_snapshot`, and `plane_filtering` statistics.

Saved calibration files include `calibration_type: "plane_3d"` and `source_modalities: ["point_cloud"]`. Old calibration files without these fields still load as `plane_3d`. Processing validates that `plane_3d` is only applied to point-cloud-compatible takes.

The belt plane normal is auto-oriented so higher Z is positive for approximately horizontal belts. If a saved calibration lacks a polygon, the belt bounding box is used as a fallback.

## Debug Visualizations

- `input_point_cloud_preview.png`: original point cloud.
- `debug_plane_segmentation.png`: background plane in gray and foreground in green.
- `debug_foreground.png`: extracted foreground only.
- `debug_clusters.png`: foreground clusters in distinct colors.

Calibrated mode additionally writes:

- `debug_calibrated_planes.png`: belt reference in cyan, ignored planes in gray, candidate foreground in green.
- `debug_belt_polygon_topview.png`: top-view belt ROI polygon in XY millimeters.
- `debug_filtered_foreground.png`: candidate object points after calibrated filtering.
- `debug_rejected_points.png`: rejected non-object points in orange.
- `debug_clusters_filtered.png`: kept clusters after calibrated object filtering.

Rendering uses a safe matplotlib path by default. Open3D offscreen rendering can be enabled with `VISION_USE_OPEN3D_OFFSCREEN=1` on systems where it is stable.

For production latency checks, run without PNG rendering:

```bash
python scripts/process_latest_real.py \
  --data-dir data \
  --calibration config/calibrations/belt_setup_2026_05_16.json \
  --skip-debug-images
```

Skipped debug stages are still listed in `result.json` profiling with a note so it is clear the renderer was intentionally bypassed.

## Profiling

Real processing writes a detailed `profiling` object in `result.json`:

```json
{
  "total_ms": 6400.0,
  "production_ms": 180.0,
  "debug_artifacts_ms": 6100.0,
  "io_ms": 45.0,
  "stages": [
    {
      "name": "dbscan_clustering",
      "category": "clustering",
      "duration_ms": 32.4,
      "input_points": 608449,
      "output_points": 18420,
      "notes": null
    }
  ]
}
```

The legacy `timing_ms` block remains for older UI/API consumers, but it is derived from profiling.

Stage categories are `io`, `preprocessing`, `calibration_filtering`, `clustering`, `measurement`, `classification`, `debug_artifacts`, and `output`. `production_ms` is the sum of processing categories excluding `debug_artifacts` and output file/state writes. `debug_artifacts_ms` is only PNG rendering. `total_ms` is wall-clock elapsed time for the run.

Use `--profile` to print the same breakdown to the console:

```bash
python scripts/process_latest_real.py \
  --data-dir data \
  --calibration config/calibrations/belt_setup_2026_05_16.json \
  --profile
```

The 200 ms target applies to `production_ms`, not debug PNG rendering. If `debug_artifacts_ms` dominates, use `--skip-debug-images` when measuring production behavior.

In live mode, throughput diagnostics are also surfaced in runtime state and result metadata:

- acquisition FPS
- processing FPS
- average/max processing latency
- queue wait time
- debug rendering and write/export overhead

When optimizing production latency, use the fast runtime cloud and skip debug images:

```bash
python scripts/process_latest_real.py \
  --data-dir data \
  --calibration config/calibrations/belt_setup_2026_05_16.json \
  --profile \
  --skip-debug-images
```

If metadata has `point_cloud_npz`, the timing table should include `load_point_cloud_fast`. With calibration, it should also include `early_calibration_crop` before voxel downsampling and outlier removal.

## POC Run Summary

`result.json` includes `poc_summary` for quick operational review:

- General: take id, timestamp, engine, calibration id/path, processing status.
- Input: point count, valid point percent, dimensions, encoder usage, capture source.
- Plane: found flag, inlier percent when available, angle, filtering enabled.
- Objects/classification: accepted/rejected objects, estimated balls/non-balls, fit and confidence summary.
- Profiling: slowest stage, total time, bottleneck category.
- Artifacts: point cloud, debug images, overlays, and result JSON presence.
- Warnings: low point count, missing plane, high rejection rate, abnormal time, suspicious calibration, and no objects detected.

The same summary prints from processing CLIs. For an existing take:

```bash
python scripts/poc_tools.py --data-dir data summary <take_id>
```

## Labeling and Evaluation

Labels are independent from `result.json` and survive reprocessing:

```bash
python scripts/poc_tools.py --data-dir data label <take_id> --label ball --label uncertain --notes "reviewed live"
```

Allowed labels are `ball`, `non-ball`, `partial ball`, `belt only`, `calibration issue`, `noisy scan`, `occluded`, and `uncertain`. They are stored at:

```text
data/takes/<take_id>/labels.json
```

Export helpers:

```bash
python scripts/poc_tools.py --data-dir data export-labels --output data/labeled_summary.csv
python scripts/poc_tools.py --data-dir data export-objects --output data/object_metrics.csv
```

Object metrics exports include diameter, sphere/ellipse fit error, point count, bounding box, confidence, filter status, and rejection reason.

## Configuration

Copy `config/processing.yaml.example` to `config/processing.yaml` and tune:

```yaml
voxel_size: 1.0
enable_outlier_removal: true
outlier_nb_neighbors: 20
outlier_std_ratio: 2.0
calibration_crop_margin_mm: 20.0
plane_distance_threshold: 1.5
plane_ransac_n: 3
plane_num_iterations: 1000
dbscan_eps: 3.0
dbscan_min_points: 30
min_cluster_points: 100
```

## Current Limitations

Objects are geometric clusters, not classified products. `class_name` is `"unknown"`, object confidence is `null`, and the summary decision is `review`. The next stages are sphere fitting, real classification, confidence calibration, and production decision rules.

## Artifact outputs

Processing outputs are now modeled as canonical artifacts (`result.artifacts`) with `stage_id`, `kind`, optional `object_id`, and metadata. Visual debug files, JSON payload references, and virtual table/metric artifacts are normalized through the same contract for Studio stage views and artifact explorer.
