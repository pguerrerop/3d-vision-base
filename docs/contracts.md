# Contracts

Stable **schemas** shared between processes. Implemented as Pydantic models in `vision_3d_acquisition.contracts` — no filesystem I/O, no sensor drivers, no queue logic.

Processes communicate on disk using these JSON shapes (`metadata.json`, `acquisition.json`, future processing manifests). Any producer or consumer may be rewritten (Python, another language) as long as it honors the same files and fields.

## Principles

- **Stable across versions** — add fields carefully; use new files or explicit schema versions for breaking changes.
- **Validate at boundaries** — acquisition validates before publish; processing validates on consume.
- **Filename indirection** — `metadata.files` lists basenames only; paths are always relative to the take folder.

## Directory contract

### `data/incoming/`

| Path | Writer | Consumer | Notes |
|------|--------|----------|-------|
| `.<take_id>.tmp/` | Acquisition | None | Hidden staging; incomplete |
| `<take_id>/` | Acquisition (via rename) | Processing | Complete after rename |
| `<take_id>/READY` | Acquisition | Processing | Must exist before consume |

### `data/processed/`

| Path | Writer | Notes |
|------|--------|-------|
| `<take_id>/` | Processing | Outputs for UI / output controller |
| `<take_id>/result.json` | Processing | Processing result contract |
| `<take_id>/input_point_cloud_preview.png` | Processing | Preview of original point-cloud input when present |
| `<take_id>/overlay.png` | Processing | Optional processed overlay |
| `<take_id>/debug_height.png` | Processing | Optional debug height image |
| `<take_id>/debug_segmentation.png` | Processing | Optional debug segmentation image |
| `<take_id>/debug_plane_segmentation.png` | Processing | Real plane/foreground segmentation |
| `<take_id>/debug_foreground.png` | Processing | Real foreground extraction |
| `<take_id>/debug_clusters.png` | Processing | Real foreground clusters |
| `<take_id>/DONE` | Processing | Completion marker |

### `data/state/`

| File | Writer | Reader |
|------|--------|--------|
| `acquisition.json` | Acquisition | UI/API, ops |
| `runtime.json` | API / workers | UI/API |
| `latest.json` | Processing | UI/API |

### `data/sessions/`

| Path | Writer | Reader |
|------|--------|--------|
| `<session_id>/metadata.json` | acquisition/live loop | API/UI/debug |
| `<session_id>/takes/<take_id>/metadata.json` | acquisition/live loop | API/UI/debug |

---

## `metadata.json`

Schema version: implicit v1 (no `schema_version` field yet; add when breaking changes occur).

Validated by `vision_3d_acquisition.contracts.metadata.AcquisitionMetadata`.

### Example (offline PLY)

```json
{
  "take_id": "2026-05-16T153012_001",
  "source": "offline_ply",
  "mode": "offline",
  "created_at": "2026-05-16T15:30:12.120Z",
  "session_id": "session_2026_05_17_120000",
  "frame_count": 1,
  "modalities": ["point_cloud"],
  "frameset": {
    "frameset_id": "2026-05-16T153012_001_fs0",
    "timestamp": "2026-05-16T15:30:12.120Z",
    "assets": {"point_cloud": "point_cloud.ply"},
    "synchronization": {"mode": "timestamp", "confidence": 0.95},
    "frame_count": 1,
    "synchronized": false,
    "timestamp_source": "file"
  },
  "files": {
    "point_cloud": "point_cloud.ply",
    "point_cloud_npz": "point_cloud.npz"
  },
  "units": {
    "x": "mm",
    "y": "mm",
    "z": "mm"
  },
  "calibration": {
    "profile_distance_mm": null,
    "x_resolution_mm": null,
    "z_scale": 1.0,
    "z_offset": 0.0
  },
  "sensor": {
    "model": null,
    "serial": null,
    "ip": null
  }
}
```

Artifact contract additions:

- `artifacts: ProcessingArtifact[]` on `ProcessingResult`
- `StageOutput.artifact_ids` optional references
- `DetectedObject.artifact_ids` optional references

Artifact normalization preserves backward compatibility: when explicit artifacts are missing, API derives stage/object artifacts from legacy `files`, `objects`, `rejected_objects`, and result summary fields.

Projection contract additions:

- Projection artifacts remain `ProcessingArtifact` entries (`kind: image`) with projection metadata fields.
- `projection_type` defines canonical view (`xy_topdown`, `xz_side`, `yz_side`, `object_crop`, `heightmap`, `depthmap`).
- `projection_coordinate_system` defines deterministic render space (origin, axes, pixel/mm, world bounds, image size, affine transform).
- Overlays should use `coordinate_space: projection_pixel` and MUST set `target_artifact_id` to a projection artifact.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `take_id` | string | yes | Unique take identifier; matches folder name |
| `source` | string or object | yes | Origin label or structured details such as USB camera index/backend |
| `mode` | string | yes | `offline` or `live` |
| `created_at` | string (ISO 8601 UTC) | yes | Creation timestamp with ms |
| `frame_count` | integer ≥ 1 | yes | Number of frames in take |
| `modalities` | string[] | no | Detected or declared input modalities |
| `session_id` | string \| null | no | Acquisition session grouping id |
| `frameset` | object | no | Frame count, synchronization, and timestamp source when known |
| `files` | object | yes | Basenames present in take folder |
| `units` | object | yes | Axis units for spatial data |
| `calibration` | object | yes | Scale/offset and profile hints |
| `sensor` | object | yes | Device identity when known |

### `files` object

| Key | Value | Description |
|-----|-------|-------------|
| `point_cloud` | string \| null | PLY filename in take folder |
| `point_cloud_npz` | string \| null | Fast runtime point-cloud NPZ |
| `height` / `heightmap` | string \| null | Height map TIFF/PNG/NPZ |
| `reflectance` | string \| null | Reflectance PNG |
| `rgb` | string \| null | RGB/2D image |
| `rgb_video` | string \| null | RGB video file, usually MP4 with AVI fallback |
| `laser_rgb` | string \| null | Laser line or overlay image |

Only non-null entries must exist on disk at publish time.

## Multi-modal inspection model

`metadata.modalities` is inferred from `metadata.files` and filenames when it is absent, so old point-cloud takes continue to load as `["point_cloud"]`. Future image or multi-sensor takes can be represented by adding only the fields that exist:

```json
{
  "take_id": "take_123",
  "modalities": ["point_cloud", "heightmap", "reflectance", "rgb", "rgb_video", "laser_rgb"],
  "frameset": {
    "frame_count": 1,
    "synchronized": false,
    "timestamp_source": "file"
  },
  "files": {
    "point_cloud": "point_cloud.ply",
    "point_cloud_npz": "point_cloud.npz",
    "heightmap": "height.tiff",
    "reflectance": "reflectance.png",
    "rgb": "rgb.png",
    "rgb_video": "rgb_video.mp4",
    "laser_rgb": "laser_overlay.png"
  }
}
```

The API returns `modalities`, `assets` grouped by modality, `frame_count`, and `frameset` for take summaries and take details.

## Runtime preview contract

Live browser preview uses one overwritten image plus metadata:

```text
data/runtime/previews/
  usb_camera_0.jpg
  usb_camera_0.json
```

API:

- `GET /api/runtime/preview` returns the latest JPEG with no-cache headers.
- `GET /api/runtime/preview/metadata` returns preview metadata.
- `GET /api/runtime/preview/{source_id}` can return a source-specific image.

Metadata example:

```json
{
  "source": "usb_camera",
  "camera_index": 0,
  "timestamp": "2026-05-17T15:10:00.120000+00:00",
  "resolution": [1920, 1080],
  "stale": false,
  "fps_estimate": 4.8,
  "path": "data/runtime/previews/usb_camera_0.jpg"
}
```

Runtime state mirrors the most important preview fields: `preview_available`, `preview_timestamp`, `preview_fps_estimate`, `preview_stale`, `preview_source`, and `preview_path`.

## Pipeline registry contract

`GET /api/pipelines` returns lightweight processing metadata:

```json
{
  "id": "3d_ball_inspection",
  "display_name": "3D Ball Inspection",
  "required_modalities": ["point_cloud"],
  "optional_modalities": ["rgb"],
  "implemented": true,
  "stages": [
    {"id": "segmentation", "display_name": "Object segmentation"},
    {"id": "classification", "display_name": "Ball classification"},
    {"id": "measurement", "display_name": "Diameter/statistics"}
  ]
}
```

Future fusion metadata can declare `required_modalities: ["point_cloud", "rgb"]` while `implemented: false`.

Registry stage descriptors are now explicit processing-unit metadata:

- `stage_id`, `display_name`, `version`, `description`
- `required_modalities`, `optional_modalities`
- `produced_artifact_kinds`
- `object_outputs`, `supports_real_time`
- `dependencies`, `optional_stage`, `condition`
- `composition` (execution order, artifact flow, conditional/optional stages)

The key Studio abstraction is: a take is a multimodal container, and a pipeline consumes a subset of that take’s modalities. This allows RGB-only, point-cloud-only, and future synchronized RGB+3D takes to share the same contracts without pretending every pipeline can consume every source.

Processed results may include `stage_outputs`, grouped by producing stage:

```json
{
  "stage": "segmentation",
  "display_name": "Object segmentation",
  "artifacts": {
    "foreground": "debug_foreground.png",
    "clusters": "debug_clusters.png"
  }
}
```

Studio derives artifact navigation from this model. Each artifact is treated as a typed engineering output with:

- `stage`: the producing stage
- `type`: image, point cloud, JSON, reference, or future placeholder
- `status`: available, missing, or future
- `filename`: optional take-relative file
- preview route: resolved through the existing take file API

Objects in `result.objects` and `result.rejected_objects` are treated as stable candidates within the take. Their `object_id` is the current UI identity used for cross-stage selection; future temporal tracking can add separate track IDs without changing this local object contract. Stage and artifact contracts must therefore avoid assuming one object, one modality, one stage, or one artifact per take.

The current artifact explorer builds typed descriptors from existing result files instead of introducing a new storage system. Future Ruler3000/Ranger3, Ranger-style stream, or GenIStream-derived outputs can map height maps, reflectance images, RGB frames, point clusters, histograms, overlays, and result payloads into the same stage/artifact descriptor shape.

### Example (USB RGB image)

```json
{
  "take_id": "2026-05-17T145500_001",
  "source": {
    "type": "usb_camera",
    "camera_index": 0,
    "backend": "AVFOUNDATION",
    "resolution": [1920, 1080]
  },
  "mode": "live",
  "created_at": "2026-05-17T14:55:00.120Z",
  "session_id": "session_usb_01",
  "frame_count": 1,
  "modalities": ["rgb"],
  "frameset": {
    "frame_count": 1,
    "synchronized": false,
    "timestamp_source": "usb_camera"
  },
  "files": {
    "rgb": "rgb.png"
  }
}
```

### Example (USB RGB video)

```json
{
  "modalities": ["rgb", "rgb_video"],
  "frame_count": 300,
  "frameset": {
    "frame_count": 300,
    "synchronized": false,
    "timestamp_source": "usb_camera"
  },
  "files": {
    "rgb": "preview.png",
    "rgb_video": "rgb_video.mp4"
  },
  "source": {
    "type": "usb_camera",
    "camera_index": 0,
    "fps": 29.97,
    "duration_seconds": 10.01
  }
}
```

### `units` object

| Key | Value | Example |
|-----|-------|---------|
| `x`, `y`, `z` | string | `"mm"` |

### `calibration` object

| Key | Type | Description |
|-----|------|-------------|
| `profile_distance_mm` | number \| null | Distance between profiles |
| `x_resolution_mm` | number \| null | X sampling |
| `z_scale` | number | Z multiplier (default 1.0) |
| `z_offset` | number | Z offset (default 0.0) |

### `sensor` object

| Key | Type | Description |
|-----|------|-------------|
| `model` | string \| null | e.g. TriSpector model |
| `serial` | string \| null | Device serial |
| `ip` | string \| null | Network address when live |

---

## `data/state/acquisition.json`

Written by `AcquisitionPublisher` after each successful publish.

### Example

```json
{
  "last_take_id": "2026-05-16T153012_001",
  "last_published_at": "2026-05-16T15:30:12.120Z",
  "last_source": "offline_ply",
  "last_mode": "offline"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `last_take_id` | string | Most recently published take |
| `last_published_at` | string | ISO 8601 UTC from metadata |
| `last_source` | string | `metadata.source` |
| `last_mode` | string | `metadata.mode` |

---

## Publish algorithm (normative)

```
function publish(take_id, metadata, file_sources):
  tmp = incoming / f".{take_id}.tmp"
  final = incoming / take_id
  assert not final.exists()
  create tmp
  for each (key, src_path) in file_sources:
    copy src_path → tmp / metadata.files[key]
  write tmp / "metadata.json" from metadata
  rename tmp → final          # atomic on same volume
  create empty final / "READY"
  write state/acquisition.json
```

---

## Python API

```python
from pathlib import Path
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher
from vision_3d_acquisition.acquisition.offline_ply import OfflinePlyAcquisition

publisher = AcquisitionPublisher(Path("data"))
take_id, folder = OfflinePlyAcquisition(publisher).acquire(
    Path("samples/pointclouds/sample.ply")
)
```

Models: `vision_3d_acquisition.contracts.metadata`.

---

## `result.json`

Validated by `vision_3d_acquisition.contracts.result.ProcessingResult`.

`processing_mode` must be explicit:

- `"mock"` means detections, decisions, and processed-output images are synthetic demonstration data.
- `"real"` means a real processing worker produced the result.

`processing_engine` is stable for real outputs:

- `"legacy"` for the compatibility segmentation path.
- `"native"` for the stage-native ball inspection path.
- `"mock"` may be used by synthetic/demo processors.

`algorithm_stage` must also be explicit:

- `"mock"` for synthetic/demo output.
- `"segmentation"` for the current real geometric pipeline.
- `"classification"` for future real classifier output.
- `"production"` for future production decision rules.

The current mock processor computes raw point-cloud stats and an input preview from the original PLY, but it does not perform real segmentation or classification.

Stable fields for POC consumers: `take_id`, `processed_at`, `processing_mode`, `processing_engine`, `algorithm_stage`, `status`, `summary`, `input_stats`, `objects`, `files`, `timing_ms`, `profiling`, `poc_summary`, and `calibration_diagnostics`.

Additional Studio-debugging fields:

- `artifacts`: typed artifact list including `kind: "overlay"` and `kind: "point_cloud"`
- `pipeline_execution`: ordered stage execution trace with status/timing/input/output/warnings/errors
- `processing_pipeline.stages`: processing-unit metadata for stage introspection

`plane_mode` is stable and explicit:

- `"calibrated"` means a saved calibration identity was part of the run.
- `"auto"` means automatic plane estimation only.
- `"disabled"` is reserved for future explicit no-plane workflows.

Experimental fields: object fit details (`diameter_mm`, `sphericity_score`, `fit_rmse_mm`) and calibration-specific `plane_filtering` internals. Compatibility image aliases such as `overlay`, `debug_height`, and `debug_segmentation` remain available for older UI consumers.

New modality-aware result fields are additive:

- `input_modalities`: modalities consumed from the take.
- `output_modalities`: produced modality/artifact classes, such as `point_cloud` and `debug_image`.
- `processing_pipeline`: pipeline name and required modalities.
- `calibration`: calibration type, source modalities, resolution source, file, and active flag.
- `session_id`: optional acquisition session id for grouping.
- `frameset_id`: optional frameset id associated with the processed result.
- `runtime_acquisition`: queue/lag/connectivity snapshot at processing time.
- `throughput`: rolling throughput metrics and warnings.
- `synchronization`: copied frameset synchronization metadata.
- `acquisition_timestamps`: acquisition and metadata timestamps.

### Overlay artifact shape

`ProcessingArtifact` now supports canonical spatial overlays:

```json
{
  "artifact_id": "segmentation_overlay_bbox_1",
  "kind": "overlay",
  "stage_id": "segmentation",
  "object_id": 1,
  "overlay_type": "bbox",
  "target_artifact_id": "foreground_clusters",
  "geometry": { "x": 120, "y": 90, "width": 80, "height": 60 },
  "style": { "stroke": "#00ff88", "fill": "rgba(0,255,136,0.08)", "line_width": 2 },
  "source_artifact_ids": ["measurement_object_1"]
}
```

Overlay spatial coordinates must declare `coordinate_space`:

- `image_pixel`: absolute image pixels relative to target artifact dimensions.
- `normalized_image`: normalized values in `[0,1]` converted to image pixels at render time.
- `plot_pixel`: pixels in the producer plot space (`plot_width`/`plot_height` metadata can be used for scaling).
- `world_mm` (future-compatible): world coordinates, non-exact on static PNGs without explicit projection.
- `point_cloud_projection` (future-compatible): requires explicit projection transform.

Rendering safety rule: if an overlay cannot be transformed reliably for its target image, Studio must mark it approximate or not render it; it must not silently draw misleading geometry.

### Stage execution report shape

```json
{
  "stage_id": "classification",
  "status": "success",
  "started_at": 1715944500.2,
  "finished_at": 1715944500.6,
  "duration_ms": 3.4,
  "warnings": [],
  "errors": [],
  "input_artifact_ids": ["foreground_clusters"],
  "output_artifact_ids": ["classification_table", "classification_overlay_ellipse_1"],
  "object_count": 2,
  "rejected_count": 1
}
```

`result.pipeline_execution` wraps these in ordered pipeline order:

```json
{
  "pipeline_execution": {
    "pipeline_id": "3d_ball_inspection",
    "stages": []
  }
}
```

### Point-cloud artifact metadata contract

Future 3D viewers consume `kind: "point_cloud"` artifacts with metadata fields:

- `coordinate_frame` (for example `camera`, `world`, `belt`)
- `units` (currently `mm`)
- `point_count`
- `bounds` (future min/max bounds)
- `projection_references` (artifact ids used for 2D overlay compatibility)

Legacy calibration fields remain: `calibration_file`, `calibration_id`, `calibration_active`, `calibration_resolution_source`, `plane_model`, `summary`, and `profiling`.

### Example

```json
{
  "take_id": "2026-05-16T153012_001",
  "processed_at": "2026-05-16T15:31:02.120Z",
  "processing_mode": "mock",
  "algorithm_stage": "mock",
  "status": "ok",
  "summary": {
    "object_count": 2,
    "ball_count": 1,
    "non_ball_count": 1,
    "decision": "review",
    "confidence": 0.84
  },
  "input_stats": {
    "point_count": 120000,
    "has_colors": false,
    "has_normals": false,
    "min_bound": [-50.0, -25.0, 0.0],
    "max_bound": [50.0, 25.0, 40.0],
    "extent": [100.0, 50.0, 40.0],
    "file_size_bytes": 8200000
  },
  "objects": [],
  "files": {
    "point_cloud": "point_cloud.ply",
    "input_preview": "input_point_cloud_preview.png",
    "overlay": "overlay.png",
    "debug_height": "debug_height.png",
    "debug_segmentation": "debug_segmentation.png"
  },
  "timing_ms": {
    "load": 40,
    "segmentation": 120,
    "classification": 35,
    "total": 210
  },
  "error": null
}
```

For real segmentation output, object classes are initially unknown:

```json
{
  "processing_mode": "real",
  "algorithm_stage": "segmentation",
  "plane_model": [0.0, 0.0, 1.0, -2.4],
  "summary": {
    "object_count": 2,
    "ball_count": 0,
    "non_ball_count": 0,
    "decision": "review",
    "confidence": null
  },
  "objects": [
    {
      "object_id": 1,
      "class_name": "unknown",
      "confidence": null,
      "point_count": 420,
      "center_mm": [12.4, 3.2, 18.6],
      "dimensions_mm": [24.1, 23.8, 22.7],
      "diameter_estimate_mm": 40.8,
      "bbox_min_mm": [0.2, -8.4, 7.5],
      "bbox_max_mm": [24.3, 15.4, 30.2]
    }
  ]
}
```
