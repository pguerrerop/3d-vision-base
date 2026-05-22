# Native 2.5D Heightmap Pipeline (mining_steel_ball_classification_25d)

## Current POC scope

- POC is currently **25D-only**.
- RGB+25D fusion is paused for timeline reasons (not removed).
- Primary runtime path:
  - TriSpector/heightmap take
  - plane detection + normalization
  - segmentation + geometry + measurement
  - 25D-only classification
  - direct publish of 25D result

## Scope and Intent
- Added a new native pipeline family `25d` with id `mining_steel_ball_classification_25d`.
- The pipeline treats captures as calibrated heightmap/range-image data, not point-cloud-only data.
- Existing 2D and 3D pipelines remain unchanged.

## Modality Semantics
- Required modalities: `heightmap`
- Optional modalities: `reflectance`, `rgb`
- Execution backend: `native`
- Internal coordinate semantics are metric (`mm`) and plane-relative after normalization.

## Canonical 2.5D Data Contract
- Added `HeightmapFrame` in `vision_3d_acquisition/vision_core/heightmap.py` with:
  - `z_mm`, `valid_mask`, optional `reflectance`
  - `x_resolution_mm`, `y_resolution_mm`
  - `origin_x_mm`, `origin_y_mm`
  - `coordinate_system`
- Serialization helpers:
  - NPZ read/write
  - metadata JSON writer
  - PNG preview generator (pseudo-color)

## Stage Architecture
Pipeline stages are integrated into the existing Stage/PipelineContext contracts with explicit belt-plane QA semantics:
1. `LoadHeightmapCapture`
2. `DetectBeltPlane`
3. `NormalizeHeightsToPlane`
4. `RemoveBeltAndSegmentObjects`
5. `FitObjectGeometry`
6. `ComputeHeightMetrics`
7. `ClassifyMiningBall25D`
8. `Generate25DOverlays`

Stage categories follow Studio semantics:
- `input`, `calibration`, `segmentation`, `geometry`, `measurement`, `classification`, `overlay`

## Belt Plane and Normalized Measurement Space
- `DetectBeltPlane` now uses a **seed -> fit -> residual-expand -> refit** strategy:
  - valid-point extraction from heightmap
  - ROI filtering
  - depth-percentile seed selection (`background_selection_mode`, `background_percentile`)
  - optional morphology + border-connected filtering on seeds
  - initial plane fit from seeds
  - residual expansion over all valid ROI pixels using `plane_background_residual_tolerance_mm` (fixed/adaptive)
  - optional plane refit from expanded inliers
- Plane equation is persisted as `ax + by + cz + d = 0`.
- Artifacts emitted:
  - `belt_plane.json`
  - `background_seed_mask.png`
  - `expanded_plane_mask.png`
  - `final_plane_inlier_mask.png`
  - `background_candidate_mask.png` (legacy alias for compatibility)
  - `background_depth_histogram.json`
  - `background_selection_debug.json`
  - `belt_plane_overlay.png`
  - `belt_plane_residuals.png`
- `NormalizeHeightsToPlane` computes canonical `height_above_belt_mm` map where `0 mm` equals belt plane.
- `RemoveBeltAndSegmentObjects` suppresses expanded-plane pixels during foreground extraction so belt residual noise does not become objects.

Depth convention semantics:
- background/belt corresponds to the **farthest** surface in ROI.
- selection mode supports:
  - `farthest_percentile`
  - `nearest_percentile`
  - `automatic` (default; inferred convention logged in debug JSON).

## Studio QA Semantics (POC)
- Stage 2 (`detect_belt_plane`) tabs:
  - `Raw heightmap`
  - `Valid mask`
  - `Plane inliers` (default)
  - `Residual heatmap`
  - `JSON`
- Stage 3 (`normalize_heights_to_plane`) tabs:
  - `Normalized height` (default)
  - `Histogram`
  - `JSON`
- Stage 4 (`remove_belt_segment_objects`) tabs:
  - `Threshold mask`
  - `Cleaned mask`
  - `Connected components`
  - `Overlay` (default)
  - `JSON`

Legacy compatibility:
- Backend class aliases are kept (`EstimateBeltPlaneStage`, `NormalizeHeightRelativeToPlaneStage`, `HeightSegmentationStage`) so older imports/runs remain viewable.

Known failure mode guards:
- explicit warnings are emitted when:
  - candidate selection is sparse or filtered out
  - inlier ratio is very low
  - residuals become extreme
  - candidate/inlier overlap is poor
  - normalized background is not near zero

## Segmentation, Geometry, and Measurement Definitions
- Segmentation is height-threshold based (`min_height_mm`, optional `max_height_mm`) with morphology and component area filtering.
- ROI support for 2.5D is available from metadata (`roi_25d`) with rectangle/polygon support, applied before segmentation.
- Footprint geometry per component includes contour, hull, ellipse-derived geometry, area, roundness/eccentricity.
- Height metrics:
  - `max_height_mm`, `mean_height_mm`, `median_height_mm`, `p95_height_mm`, `height_std_mm`
- Volume proxy:
  - `sum(max(height_above_belt_mm, 0) * pixel_area_mm2)`
- Deformation-oriented features:
  - height asymmetry
  - flatness
  - eccentricity
  - edge roughness
  - local curvature proxy

## Classification Semantics
- Initial classifier is heuristic and threshold-based, architecture-ready for future ML.
- Domain labels are stored as detailed labels while preserving existing `class_name` compatibility (`ball`/`non_ball`/`unknown`).
- Supports classes and subtypes aligned with requested taxonomy:
  - `Bola buena`
  - `Scrap de Bola` variants
  - `Chatarra` variants

## Overlay and Artifact Contracts
- Added explicit 2.5D overlays/artifacts consistent with existing explorer/inspector flows:
  - `heightmap`
  - `normalized_heightmap`
  - `belt_plane`
  - `height_segmentation`
  - `height_overlay`
  - `measurement_overlay`
  - `classification_overlay`
- Result `files` contract was extended for heightmap-native outputs.

## Calibration and Encoder Metadata Rationale
- Added `heightmap_calibration` stage artifact with metric scaling and coordinate system context.
- Existing calibration model retains shared `PlaneCalibration`; 2.5D-specific runtime metadata is injected in pipeline artifacts.
- Encoder-aware metadata is now preserved in the pipeline context/result artifacts:
  - `encoder_ticks_per_mm`
  - `scan_direction`
  - `profile_distance_mm`
  - `belt_speed_mm_s`
- Full acquisition synchronization is intentionally deferred.

## Studio Integration
- Pipeline is discoverable through existing pipeline registry and API execution route.
- No special UI path was added; artifacts/stages remain compatible with current Studio tabs, explorer, and JSON rendering.

## Future Fusion Readiness
- Contracts are isolated by modality and stage outputs so RGB + 2.5D fusion can be added as a later pipeline stage.
- Current design keeps acquisition backend replaceable and replay/offline friendly.

## Studio Integration and Validation Workflow

### Studio stage-native views for `mining_steel_ball_classification_25d`
When this pipeline is selected, Studio stage views are exposed through the existing semantic tab system (no special UI path):
- `Heightmap`
- `Normalized Height`
- `Belt Plane`
- `Segmentation`
- `Measurements`
- `Classification`
- `Overlays`
- `Artifacts`
- `JSON`

### 25D preview artifacts rendered in Studio
The following artifact previews are produced and visible through current artifact/viewer components:
- raw heightmap preview (`heightmap`)
- normalized heightmap preview (`normalized_heightmap`)
- belt plane residuals (`belt_plane_residuals`)
- threshold/cleaned segmentation masks (`threshold_mask`, `cleaned_mask`)
- segmentation overlay (`height_segmentation` / `segmentation_overlay.png`)
- measurement overlay (`measurement_overlay`)
- classification overlay (`classification_overlay`)

### Inspector 25D diagnostics
Inspector now surfaces:
- plane coefficients and residual statistics (from `belt_plane` metadata)
- height threshold configuration (from `height_segmentation` metadata)
- connected component count (from `connected_components` metadata)
- per-object 25D measurement fields including height stats, footprint/volume proxy, and class label/group hints

## Synthetic 25D data generation
Create synthetic offline take(s):

```bash
python scripts/create_synthetic_25d_take.py --data-dir data --session-id synthetic_25d_demo
```

Synthetic scenes include:
- tilted belt plane
- one round ball-like object
- one flattened/deformed ball-like object
- one elongated scrap-like object
- invalid/no-return regions
- optional reflectance

## 25D demo runner
Run end-to-end 25D processing demo:

```bash
python scripts/run_25d_pipeline_demo.py --data-dir data --take-id <take_id>
```

or auto-create + run:

```bash
python scripts/run_25d_pipeline_demo.py --data-dir data
```

The script prints:
- result path
- pipeline id/family/status
- object count
- plane residual stats
- per-object class/height/volume summary
- artifact paths

## API processing validation
`POST /api/takes/{take_id}/process` with:

```json
{
  "pipeline_id": "mining_steel_ball_classification_25d",
  "reprocess": true
}
```

returns:
- `pipeline_id`
- `pipeline_family`
- `status`
- `result_path`
- `artifacts` (id/path list)
- `result` payload including 25D measurements/artifacts

## Current limitations
- classifier is heuristic (not ML)
- no RGB+25D fusion implemented yet
- segmentation parameter live-preview remains optimized for process-service 2D flows

## Future fusion step
A future fusion stage should consume:
- RGB defect cues (chips/cracks)
- 25D deformation/volume cues
and produce final multimodal classification under the existing stage/artifact contracts.

### API contract validation helper
Direct API-function validation script:

```bash
python scripts/run_25d_api_validation.py --data-dir data --take-id <take_id>
```

This invokes `process_take_for_pipeline(...)` with:
- `pipeline_id = mining_steel_ball_classification_25d`

## Plane normalization convention

- `EstimateBeltPlane` models belt/background plane from valid height pixels.
- `NormalizeHeightRelativeToPlane` uses:
  - `height_above_belt_mm = raw_z_mm - plane_z_at_xy`
- Expected behavior:
  - background near `0 mm`
  - above-belt objects positive
  - invalid/no-return remain invalid
  - optional negative clipping configurable

## Required plane-debug artifacts

- `raw_heightmap_preview.png`
- `valid_mask.png`
- `plane_fit_roi_mask.png`
- `plane_inlier_mask.png`
- `plane_residuals.png` (`belt_plane_residuals.png`)
- `normalized_heightmap_preview.png` (`normalized_heightmap.png`)
- `normalized_height_histogram.json`
- `plane_fit_debug.json`

## Known cube scale validation workflow

- Optional stage: `ValidateKnownObjectScale25D` (disabled unless `metadata.known_object_25d.enabled=true`).
- Uses cube known dimensions to validate measurement scale after normalization.
- Outputs:
  - measured width/depth/height
  - scale errors by axis
  - pass/fail per axis
  - recommended scale corrections
- Default behavior is read-only diagnostics (no automatic calibration mutation).

## Direct 25D publication path

- `POST /api/takes/{take_id}/publish-25d-result`
- Publishes latest successful 25D take result into `PublishedInspectionResult` with:
  - `source_refs.result_mode = "25d_only"`
  - 25D-only source refs and overlays

## Validation script

```bash
python scripts/validate_25d_cube_calibration.py --data-dir data --take-id <take_id> \
  --known-width-mm <value> \
  --known-depth-mm <value> \
  --known-height-mm <value>
```

Prints:

- plane fit status + inlier ratio
- residual stats
- normalized background near-zero stats
- measured cube dimensions
- scale errors + recommended scale corrections
- artifact paths

## Current limitations

- Cube validation currently recommends corrections only.
- No automatic calibration write-back in this step.
- Fusion and operator-level fusion publication remain paused for POC scope.
- `reprocess = true`

and prints the full response payload.

## Low-gradient reference-surface strategy (POC default)

- New default in `DetectBeltPlaneStage`:
  - `background_detection_strategy = "low_gradient_surface"`
- Motivation:
  - Percentile-only seed selection was too patchy on real TriSpector captures.
  - The POC now prefers broad low-gradient depth regions as reference-surface candidates.
- Workflow:
  - Compute depth-gradient magnitude on valid pixels.
  - Threshold into low-gradient mask (`fixed | percentile | otsu`).
  - Connected-components candidate scoring with area + constancy + border-touch priors.
  - Select best reference-surface component.
  - Fit model in `auto | plane | constant_z`.
    - `auto` falls back to `constant_z` when plane quality is poor.
- New/updated artifacts:
  - `depth_gradient_magnitude.png`
  - `low_gradient_mask.png`
  - `low_gradient_components_overlay.png`
  - `low_gradient_components.json`
  - `reference_surface_selected_mask.png`
  - `reference_surface_candidates.json`
  - `gradient_debug.json`
- Segmentation now suppresses selected reference-surface pixels before final object mask generation.

## Studio-tunable 25D reference-surface parameters

The `Detect belt plane` stage now accepts run-time tuning parameters from Studio (`stage_params`) so engineers can iterate without code edits:

- `background_detection_strategy`: `depth_percentile_plane | low_gradient_surface`
- `gradient_smoothing_kernel`
- `gradient_threshold_mode`: `fixed | percentile | otsu`
- `gradient_threshold_value`
- `gradient_threshold_percentile`
- `low_gradient_morphology_enabled`
- `low_gradient_open_kernel`
- `low_gradient_close_kernel`
- `low_gradient_fill_holes`
- `low_gradient_min_component_area`
- `reference_surface_selection_mode`
- `reference_surface_model`: `plane | constant_z | auto`
- `reference_surface_max_plane_residual_p95_mm`
- `object_min_height_mm` (mapped to segmentation min-height threshold)

Model selection diagnostics explicitly report:

- `reference_surface_model_type`
- `plane_coefficients` or `constant_z_mm`
- `selected_component_id`
- `selected_component_area_ratio`
- `model_residual_mean_mm`
- `model_residual_p95_mm`
- `model_selection_reason`
