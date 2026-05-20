# Diagnostics and Studio Workflow

## Operations Workflow

Use Operations for the current production run. Demo mode enlarges the main KPIs, keeps the latest inspection image visible, auto-refreshes through the existing event stream, and hides lower-value context for presentations.

Check:

- decision and object counts
- processing speed
- calibration validity
- latest warning
- raw input assets and debug image presence
- calibration type and modality compatibility

## Studio Workflow

Use Studio to review more than one take and reason through Input → Segmentation → Classification → Measurements → Fusion. Studio shows sessions, takes, selected pipeline, compatibility, stage timing, and stage artifacts.

Available review controls:

- sort by newest
- sort by slowest
- sort by warning count
- filter failed only
- filter warnings only
- filter by acquisition session

Select a take to inspect source state, pipeline compatibility, stage outputs, modality tabs, processed artifacts, object tables, profiling, POC summary, and full `result.json`.

## Diagnostics Workflow

Use Diagnostics to answer what is broken, slow, stale, or malformed. Diagnostics exposes runtime state from `runtime.json`: acquisition connectivity, queue size, lag, throughput, live warnings for overload/staleness, preview freshness, profiling, metadata, result payloads, and raw JSON.

## Multi-modal Inspection Model

Debug views show the take as a set of input modalities, not as a point cloud only. Available raw-input tabs are derived from API `modalities` and grouped `assets`:

- Point cloud
- Heightmap
- Reflectance
- RGB
- RGB video
- Laser image
- Metadata

Only available tabs are shown. Missing assets use empty states instead of implying a point cloud should exist. RGB video takes show the recorded video asset plus the captured preview image. Current processing still requires `point_cloud`, so image-only takes are useful for contract and UI validation but are not accepted by the ball inspection pipeline yet.

USB camera captures update runtime diagnostics with `acquisition_source: "usb_camera"`, camera index, connection state, latest frame timestamp, session, acquisition FPS when measured, preview timestamp, preview FPS estimate, preview availability, and stale state. Operations, Studio, Calibration, and Diagnostics surface those fields without adding streaming video infrastructure.

## Browser Live Preview

The web UI polls `GET /api/runtime/preview/metadata` and refreshes `GET /api/runtime/preview` with cache-busting query strings. The latest preview is a single overwritten JPEG, not a stream. Visual states are:

- `CONNECTED`: acquisition is connected and the preview metadata is fresh.
- `STALE`: the last preview exists but is older than the freshness threshold.
- `DISCONNECTED`: runtime reports acquisition disconnected.
- `NO PREVIEW AVAILABLE`: no preview frame has been exported yet.

The native OpenCV window is still available with `scripts/capture_usb_camera.py --preview-window` for engineering work, but it is not required for operator or calibration workflows.

## Studio

Studio is the place for stage-by-stage experimentation and validation. It is not the Operations surface and not Calibration. It supports take/session selection, pipeline selection, compatibility display, stage listing, stage timing, and stage artifact viewing. Future RGB and fusion pipelines are visible as disabled metadata so the UI shape is ready without implying they are implemented.

Studio now behaves like a workstation rather than a scrolling dashboard:

- left data browser stays available for sessions, takes, filters, and pipeline compatibility
- center workbench switches by modality tab or selected stage
- right inspector follows selected stage, object, and artifact context
- panes scroll independently inside the fixed application shell

Stage selection is expected to change the workbench meaningfully: Segmentation focuses on masks/clusters/candidate objects, Classification on labels/confidence/rejection reasons, Measurements on tables/statistics/tolerances, and Fusion on future RGB/3D synchronization and alignment. Object selection and artifact selection are shared UI context, so engineers can move between stages without losing the object or output they are investigating.

### Spatial inspection workflow

Studio object debugging is now spatial-first:

- selecting an object highlights matching overlays on image artifacts
- hovering object rows highlights corresponding overlays
- hovering/clicking overlays highlights/selects the object row
- artifact explorer previews render canonical overlay types (`bbox`, `ellipse`, `centroid`, `polyline`, `text`)

Overlay targeting and transform checks:

- overlays render only when `target_artifact_id` resolves to a visible image artifact
- missing targets are surfaced as `No renderable target artifact available.`
- coordinate transforms are deterministic for `image_pixel`, `normalized_image`, and `plot_pixel`
- `world_mm` overlays on static debug images are explicitly marked approximate
- `projection_pixel` overlays are preferred and expected for canonical projection artifacts
- overlays without `target_artifact_id` are rendered in compatibility mode with warnings

Diagnostics remains the place for full runtime internals, queue/FPS troubleshooting, logs/events, profiling dumps, and raw payload inspection.

### Execution introspection workflow

Use Studio execution diagnostics to debug flow, not just end results:

- execution graph shows ordered stage flow with status (`success`, `warning`, `failed`, `skipped`)
- stage inspector shows timings, warnings/errors, and input/output artifact counts
- artifact inspector shows producer stage and lineage chain
- object inspector shows generating stages and related artifacts for provenance

### Overlay debug panel

Studio inspector includes compact overlay debugging details:

- `target_artifact_id` and resolved target title
- `coordinate_space`
- raw geometry payload
- transformed SVG geometry
- renderable yes/no
- warnings and approximate status

## Labeling Workflow

Labels live outside processing outputs:

```text
data/takes/<take_id>/labels.json
```

Example:

```json
{
  "take_id": "take_001",
  "labels": ["ball", "uncertain"],
  "notes": "partial occlusion near belt edge",
  "reviewer": "operator",
  "updated_at": "2026-05-16T18:00:00Z"
}
```

CLI:

```bash
python scripts/poc_tools.py --data-dir data label take_001 --label ball --label uncertain --notes "partial occlusion"
python scripts/poc_tools.py --data-dir data list-labels
```

## Calibration Validation

Calibration diagnostics are written into `result.json` and summarized in `poc_summary`. They flag excessive plane tilt, low plane inlier percentage, abnormal point density, suspicious scaling, unusual offsets, and missing encoder reporting when calibration is active.

Active calibration now includes compatibility status (`compatible`, `warning`, `incompatible`), confidence score, age, and recommendation support based on take modalities and recent compatible calibrations.

Use these signals to identify bad mounting, conveyor movement, bad calibration, and encoder/capture problems before trusting classification output.

## Exports

Dataset summary:

```bash
python scripts/poc_tools.py --data-dir data export-labels --output data/labeled_summary.csv
```

Object metrics:

```bash
python scripts/poc_tools.py --data-dir data export-objects --output data/object_metrics.csv
```

Object metrics include take labels, object id, accepted/rejected status, class, confidence, diameter, fit error, point count, bounding box, dimensions, filter status, and rejection reason.

## Contract Validation

Validate a processed result:

```bash
python scripts/poc_tools.py --data-dir data validate-result <take_id>
```

Stable fields include `take_id`, `processed_at`, `processing_mode`, `processing_engine`, `algorithm_stage`, `status`, `summary`, `input_modalities`, `output_modalities`, `processing_pipeline`, `pipeline_execution`, `input_stats`, `objects`, `files`, `artifacts`, `timing_ms`, `profiling`, `poc_summary`, and `calibration_diagnostics`.

Experimental fields remain the object fit details and calibration-specific plane filtering diagnostics. Deprecated compatibility fields should remain present until frontend/API consumers no longer use them.

Artifact checks:

- Verify `result.artifacts` is non-empty for processed takes.
- Confirm stage/object links via `stage_id` and `object_id`.
- Confirm no duplicate `artifact_id` values when explicit and derived artifacts coexist.
