# Processing Units Model

A processing unit is a stage-level reusable computation block. It consumes pipeline context (modalities, prior artifacts, calibration, objects), and produces artifacts, objects, metrics, warnings, and result fields that Studio can inspect.

## Required metadata

- `stage_id`
- `display_name`
- `version`
- `description`
- `required_modalities`
- `optional_modalities`
- `produced_artifact_kinds`
- `object_outputs`
- `supports_real_time`
- `dependencies`
- `optional_stage`
- `condition`

## Implementation pattern

1. Read inputs from `PipelineContext`.
2. Validate required modalities and upstream artifacts.
3. Write outputs back to context.
4. Register artifacts with `context.add_processing_artifact(...)`.
5. Register warnings/metrics/timing in context/profiler metadata.

## Output rules

- Every visual/debug output must be registered as an artifact.
- Every summary/table must be explicit or derivable by artifact normalization.
- Object-level outputs must include `object_id`.
- Stage outputs must be inspectable in Studio via `artifacts`.

## Canonical artifact contract

```json
{
  "artifact_id": "classification_table",
  "stage_id": "classification",
  "object_id": null,
  "kind": "table",
  "title": "Classification table",
  "description": "Per-object ball/non-ball classification output.",
  "path": null,
  "mime_type": "application/json",
  "preview_available": false,
  "created_at": "2026-05-17T00:00:00Z",
  "metadata": {
    "source": "native_pipeline",
    "producer": "BallClassificationStage",
    "modality": "point_cloud",
    "derived_from": ["segmentation"]
  }
}
```

## Canonical overlay artifact contract

```json
{
  "artifact_id": "classification_overlay_ellipse_1",
  "kind": "overlay",
  "stage_id": "classification",
  "object_id": 1,
  "overlay_type": "ellipse",
  "target_artifact_id": "foreground_clusters",
  "geometry": {
    "cx": 220,
    "cy": 180,
    "rx": 50,
    "ry": 48,
    "rotation_deg": 10
  },
  "style": {
    "stroke": "#4f8cff",
    "fill": "rgba(79,140,255,0.12)",
    "line_width": 2,
    "label": "ball 72%"
  },
  "source_artifact_ids": ["classification_object_1"]
}
```

Coordinate-space guidance for processing units:

- Always set `target_artifact_id` to a real image artifact id.
- Always set `coordinate_space` explicitly.
- Emit `image_pixel`, `normalized_image`, or `plot_pixel` only when deterministic 2D transforms are available.
- If only `world_mm` is available for a static PNG without calibrated projection, mark `approximate: true` and include warning text:
  - `Overlay is approximate: world coordinates projected onto static debug image.`

## Projection-Producing Processing Units

Projection artifacts are first-class artifacts, not ad hoc screenshots. They provide stable renderable coordinate systems for overlays and future interaction.

Required projection fields:

- `projection_type`: `xy_topdown | xz_side | yz_side | object_crop | heightmap | depthmap`
- `projection_coordinate_system`: origin, axis directions, pixel/mm, world bounds, image size, optional affine transform
- `projection_metadata`: rendering style, depth range, source references
- `projection_transform_id`

Overlay targeting rules:

- Overlays MUST target projection artifacts through `target_artifact_id`.
- New units should emit `coordinate_space: projection_pixel` for 2D overlays on projection artifacts.
- Old `plot_pixel` and missing-target overlays remain supported only in compatibility mode with warnings.

Why debug screenshots are insufficient:

- camera angle and axis scaling are not stable
- pixel coordinates are not deterministic across renders
- overlays appear plausible but are not engineering-accurate

Recommended unit outputs:

- raw artifacts
- projection artifacts
- overlays bound to projection targets
- measurements
- execution diagnostics and transform warnings

## Object visualization contract by stage

- Segmentation emits object regions (`bbox`, optional `polyline` contours) targeting segmentation image artifacts.
- Classification emits shape-fit overlays (`ellipse`) and confidence labels (`text`) targeting the same base image.
- Measurement emits provenance annotations (`centroid`, text labels) for dimensional debugging.
- Overlay types currently rendered in Studio SVG overlays: `bbox`, `ellipse`, `centroid`, `polyline`, `text`.

## Example units

- `BeltObjectSegmentationUnit`
- `BallClassificationUnit`
- `DiameterMeasurementUnit`
- `FutureFusionUnit` (placeholder until fusion is implemented)

## Pipeline composition philosophy

- Pipelines are assembled from reusable units instead of monolithic algorithm blocks.
- Composition is explicit: `dependencies`, modality requirements, and artifact flow are documented in the registry.
- Optional and conditional units are represented in metadata before they are implemented.
- Execution remains in-process and ordered; no distributed scheduler, queue, or database is introduced.

## Example skeleton

```python
class ExampleUnit:
    stage_id = "classification"

    def run(self, context: PipelineContext) -> None:
        objects = context.require_artifact("objects")
        # compute
        context.set_artifact("objects", objects)
        context.add_processing_artifact(
            artifact_id="classification_table",
            stage_id="classification",
            kind="table",
            title="Classification results",
            metadata={"producer": self.__class__.__name__},
        )
```

## Testing rules

- Unit tests for required modality validation.
- Tests for artifact registration and serialization.
- Compatibility tests for old `result.json` payloads.
- Tests for derived artifact fallback and no duplication.

## UI contract

- Studio discovers stage outputs from normalized `result.artifacts`.
- Inspector reads selected artifact metadata and object links.
- Object/artifact linking is bidirectional via `artifact.object_id`.
