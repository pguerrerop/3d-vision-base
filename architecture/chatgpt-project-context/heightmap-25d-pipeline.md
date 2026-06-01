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
  - ROI filtering (reference-surface estimation only)
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
- ROI semantics for 2.5D:
  - ROI is the trusted search region for reference-surface detection only.
  - ROI affects low-gradient computation, candidate selection, plane/reference fitting, and residual expansion.
  - ROI does not crop normalized output, segmentation output, object measurements, or classification overlays.
  - Normalization and segmentation continue on the full valid frame.
  - ROI types are explicit and extensible: `rectangle`, `polygon`, `vertical_band`.
  - `vertical_band` exists for conveyor workflows where scan/image height changes between takes due to encoder/trigger timing; it constrains only X-range and always rasterizes as full-height.
  - `vertical_band` serialization:
    - `{"type":"vertical_band","x":120,"width":340}`
    - runtime mask normalization resolves to `y=0,height=image_height` while preserving `type` in debug/config metadata.
- Studio includes visual ROI editing for reference-surface tuning (raw heightmap, depth gradient, low-gradient mask, selected surface), with `Apply ROI` / `Reset to full-frame`.
- Recommended workflow: set ROI on conveyor/background -> tune low-gradient params -> verify selected surface -> verify normalized height -> verify segmentation.
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

### Execution order (conservative sph3d refinement)
Classification runs in two passes per object:

1. **Primary heuristic classifier** (`_classify_25d` in `stages_25d.py`) — unchanged thresholds for good-ball detection:
   - height, 3D sphericity, eccentricity, flatness, edge roughness, volume
   - objects matching good-ball rules become `bola_buena` / `BALL_GOOD`
2. **Secondary sph3d fallback** (`apply_sphericity_3d_fallback_to_object` in `classification_superclass.py`) — runs only when primary result is **not** `BALL_GOOD` or `SCRAP_METAL`

Good-ball objects return immediately; fallback logic is never applied to them.
Primary `SCRAP_METAL` results (e.g. `planchuela` from low height/volume) are also preserved.

### sph3d fallback thresholds (calibration-oriented)
Uses only `feature_sphericity_3d` (`sph3d`), while the old `sphere_fit` display has been renamed to `feature_footprint_roundness` because it is only a 2D ellipse-axis ratio:

| Condition | Label | Superclass |
|-----------|-------|------------|
| `sph3d < 0.30` | `chatarra` | `SCRAP_METAL` |
| `0.30 <= sph3d < 0.75` | `bola_con_chip` | `BALL_SCRAP` |
| `sph3d >= 0.75` | unchanged (primary result) | unchanged |

Thresholds are intentionally heuristic, derived from observed data, and meant for iterative calibration — not ML or multi-feature fusion.

Each non-good object receives debug metadata on the result payload:

```json
{
  "debug": {
    "sph3d_rule": {
      "sph3d": 0.42,
      "threshold_scrap_metal": 0.30,
      "threshold_ball_scrap": 0.75
    }
  },
  "classification_reason": "BALL_SCRAP because 0.30 <= sph3d=0.420 < 0.75"
}
```

Superclass aggregation (`select_dominant_classification`) is unchanged.

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

## Height Semantics Contract (Canonical)
- Canonical semantic fields:
  - `raw_sensor_z`: decoded sensor Z (little-endian 16-bit reconstruction); higher numeric means higher decoded Z.
  - `plane_signed_distance`: signed distance to fitted belt plane.
  - `height_above_belt`: authoritative physical measurement (`0 = belt`, positive above belt).
  - `preview_normalized`: display-only normalization field, never used for measurement/classification.
- Every height-related artifact metadata now carries:
  - `semantic_field`, `units`, `value_min`, `value_max`
  - `color_scale_min`, `color_scale_max`, `color_map`
  - `positive_direction`, `is_measurement_authoritative`
  - `source_artifact_id`, `stage_id`
- Studio contract:
  - Legend labels and colorbar min/max are sourced from artifact metadata.
  - Hover values come from the active semantic field of the selected view.
  - Legacy previews with missing semantic metadata must be marked as unknown semantics.
- Processing contract:
  - Segmentation/measurement/classification consume `height_above_belt`.
  - `preview_normalized` is blocked from measurement/classification paths.
  - Height metrics explicitly carry `height_metrics_semantic_field = "height_above_belt"`.
- Parser contract:
  - Keep little-endian `uint16` reconstruction as default.
  - `parser_metadata.json` includes effective bit-depth diagnostics (`raw min/max`, `p01/p99`, `unique count`, warnings for possible packed 12/14-bit ranges).
The following artifact previews are produced and visible through current artifact/viewer components:
- raw heightmap preview (`heightmap`)
- normalized heightmap preview (`normalized_heightmap`)
- belt plane residuals (`belt_plane_residuals`)
- threshold/cleaned segmentation masks (`threshold_mask`, `cleaned_mask`)
- segmentation overlay (`height_segmentation` / `segmentation_overlay.png`)
- measurement overlay (`measurement_overlay`)
- classification overlay (`classification_overlay`)

### Canonical numeric source contract (2.5D)
- Numeric processing stages read canonical numeric sources only, in priority: `height16.tif` -> `heightmap_frame.npz` -> `normalized_heightmap.npz`.
- Preview images (`heightmap_preview.png`, `normalized_heightmap.png`, overlays) are `display_only=true` and are never valid computation inputs.
- Numeric artifacts/metadata are marked `numeric_source=true`.
- Normalized hover, histogram, and legend share one numeric source (`normalized_heightmap.npz`) with transform metadata from `normalized_heightmap_display.json`.

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
# Engineering Observability Extensions

## Semantic Height Visualization

Engineering stage views now support explicit semantic modes over existing artifacts:

- `Absolute Z`
- `Height Above Belt` (default engineering interpretation)
- `Normalized Height`
- `Residual To Plane`
- `Sphere Fit Residual` (contract-ready; populated when produced)

Color semantics are renderer-configurable with:

- `auto`
- `symmetric around zero`
- `manual min/max`
- `percentile clipping`
- optional outlier saturation

## Residual Artifact Contracts

`detect_belt_plane` publishes residual-focused diagnostics in additive form:

- `belt_plane_residuals` (heatmap)
- `plane_residual_histogram` (json histogram)
- existing ROI/candidate/inlier masks remain the source of plane-fit explainability.

## Object Provenance and Geometry Diagnostics

Per-object 25D rows include:

- `measurement_provenance` (bbox/coverage/contour/point lineage)
- `sphere_fit_diagnostics` (estimated radius/diameter/truncation/confidence)
- `classification_explanation` (metric contributions + rejected-ball reasons)

This preserves current stage contracts while making classification and geometric behavior explainable.

## Second-pass canonical height semantics

This pipeline now treats height semantics as an explicit architectural layer.

### Canonical geometry domains

- `raw_sensor_z`: acquisition/debug domain (sensor-space heights).
- `plane_signed_distance`: intermediate geometry domain (signed residual to fitted belt plane).
- `height_above_belt`: canonical physical geometry domain used by measurements, classification, reports, exports, and default overlays.

### Semantic transform chain

- `raw_sensor_z` is produced by `raw_decode`.
- `plane_signed_distance` is produced from `raw_sensor_z` by `signed_distance` (plane-aware residual transform).
- `height_above_belt` is produced from `plane_signed_distance` by `invert_signed_distance`.
- Display-only `preview_normalized` is produced from canonical semantic rasters via `normalization`; it is never a measurement source.

All major height artifacts now include transform lineage metadata:

- `semantic_field`
- `derived_from`
- `transform` (`type`, and optional transform details like `plane_id`)

### Numeric raster contract

The pipeline persists canonical semantic numeric rasters:

- `raw_sensor_z.values.f32`
- `plane_signed_distance.values.f32`
- `height_above_belt.values.f32`

Each semantic raster carries metadata for runtime-safe consumption:

- `semantic_field`
- `coordinate_space`
- `units`
- `dtype`
- `source_stage`

### Semantic-first artifact model

Consumers should resolve artifacts by `(semantic_field, representation)` rather than filename. This is additive and backward-compatible with legacy file names.

### Hover/render consistency guarantee

Hover sampling and numeric diagnostics consume semantic rasters only. Colorized previews are visualization artifacts and are not used as numeric sources.

### Canonical color mapping contract (`HeightColorMapping`)

To eliminate any visual disagreement between the rendered preview, the legend, hover sampling and debug reconstructions, all height visualization consumers share a single canonical contract:

```
{
  "semantic_field": "height_above_belt",
  "units": "mm",
  "value_min": <raw data min>,
  "value_max": <raw data max>,
  "color_scale_min": <LUT lower bound>,
  "color_scale_max": <LUT upper bound>,
  "color_map": "turbo" | "viridis" | "magma" | "gray" | <id>,
  "direction": "higher_is_hotter" | "lower_is_hotter",
  "clamp": true,
  "source": "artifact_metadata" | "render_context" | "legacy_fallback"
}
```

Rules:

- the LUT is always anchored data-natively (`value_min` → cool, `value_max` → hot); `direction` only changes label/tick orientation in the legend.
- backend writes a `color_mapping` block on the rendered preview artifact, on the display metadata JSON, on the render context artifact and on the canonical numeric raster artifact (`height_above_belt_raster`).
- frontend resolution priority: explicit `color_mapping` block → render context (`render_vmin/vmax`, `colormap_id`) → display metadata (`color_scale_min/max`, `color_map`) → legacy fallback (warning).
- `HeightLegend` renders the colorbar gradient from the same named LUT (`turbo`/`viridis`/`magma`/`gray`) the image renderer uses; no hard-coded CSS gradients.
- hover sampling and reconstruction diffs use the same `colorMap` name + `colorScaleMin/Max`, so scalar→RGB and RGB→t inferences are LUT-consistent.

## Advanced Geometric Ball-Quality Semantics (v2 additive)

The `measurement` + `classification` stages now expose physically-grounded, semantically grouped ball-quality metrics while preserving legacy flat fields.

### Semantic metric groups

Per-object payloads now include:

- `footprint_geometry`
- `surface_geometry`
- `sphere_consistency`
- `damage_metrics`

These are additive and mirrored into `classification_explanation.metrics_used` and `semantic_group_summaries`.

### Coordinate/space assumptions

- `footprint_geometry` operates on XY contour/mask geometry in metric space (mm via calibrated resolutions).
- `surface_geometry` and `damage_metrics` operate on visible 2.5D cap points (`height_above_belt`) mapped to metric XYZ.
- `sphere_consistency` compares visible heights against fitted sphere-cap expectations; hidden hemisphere is inferred, never directly observed.

### Key metric definitions

- Circularity: `C = 4*pi*A / P^2` (`A`: contour area in mm², `P`: perimeter in mm).
- Radial boundary uniformity: radial mean/std/CV/max deviation from contour centroid.
- Sphere fit: least-squares fit over visible XYZ points; emits center/radius/RMSE/max error.
- Ellipsoid proxy fit: PCA-based axes/aspect + residual proxy for deformation separation.
- Sphere-vs-ellipsoid gain: normalized RMSE improvement from sphere to ellipsoid proxy.
- Radial height residual: RMSE/max error between observed visible cap and expected spherical cap.
- Surface completeness: fraction of visible object support with non-trivial above-belt occupancy.
- Volume coherence: observed volume proxy vs expected sphere volume from fitted radius.
- Damage proxies: roughness (residual stats), flat-region ratio/largest patch, discontinuity and high-curvature ratios.

### Classification/explanation integration

- Heuristic classification now consumes legacy signals plus grouped evidence (`radial_cv`, sphere-fit residual, radial-height residual, completeness, flat/discontinuity evidence).
- Explanations now include:
  - group-level pass/fail summaries (`semantic_group_summaries`)
  - per-metric contribution rules spanning footprint/surface/consistency/damage families
  - preserved legacy rule compatibility.

### Studio integration

- Inspector now renders compact grouped summaries for:
  - Footprint geometry
  - Surface geometry
  - Sphere consistency
  - Damage metrics
- Classification rule view renders semantic group pass/fail summaries above per-rule tables.

### Limits and caveats

- Ellipsoid fitting is currently a robust PCA proxy, not a full nonlinear algebraic ellipsoid fit.
- Sphere/consistency metrics use only visible cap geometry; occluded/base-side damage is inferred probabilistically.
- Flat/discontinuity metrics are local geometric proxies and should be combined with future RGB crack cues.

### Future-readiness hooks

- Grouped metrics are isolated for multimodal fusion, ML feature export, calibration, and dataset analytics without refactoring stage contracts.

## Synthetic Geometry Validation Harness

An additive synthetic validation framework now exists to stress-test metric-family discrimination under controlled physical defect modes.

### Generator scope

- Module: `vision_3d_acquisition/synthetic/advanced_geometry_25d.py`
- Supported synthetic object types:
  - `good_ball`
  - `worn_ellipsoid`
  - `truncated_sphere`
  - `chipped_sphere`
  - `flattened_ball`
  - `elongated_scrap`

### Acquisition realism controls

Per sample, configurable:

- gaussian height noise
- missing/no-return regions
- belt-plane tilt
- edge aliasing perturbation

All runs are seed-driven and deterministic.

### Persisted synthetic metadata contract

Each generated take metadata includes:

- `synthetic_object_type`
- `generation_parameters`
- `expected_failure_modes`
- `expected_metric_family_reactions`
- acquisition-effects configuration

Ground-truth debug assets are also persisted in take folders (`synthetic_gt_mask.png`, `synthetic_flat_regions_gt.png`).

### Discrimination benchmark runner

- Script: `scripts/run_advanced_25d_geometry_validation.py`
- Generates parameterized synthetic takes, runs the existing `mining_steel_ball_classification_25d` pipeline, and writes:
  - JSON report (`data/reports/advanced_25d_metric_family_report.json`)
  - Markdown summary (`data/reports/advanced_25d_metric_family_report.md`)
  - CSV feature dataset (`data/reports/advanced_25d_geometry_validation.csv`)
  - Aggregate analytics JSON (`data/reports/advanced_25d_geometry_analysis.json`)

### Classifier-studio feature export alignment

The aggregate analytics JSON now includes a lightweight classifier-ready schema:

- `feature_schema_version: "v1"`
- `feature_groups` grouped as:
  - `footprint_geometry`
  - `surface_geometry`
  - `sphere_consistency`
  - `damage_metrics`
- `feature_metadata` per feature:
  - semantic group
  - units
  - directionality (`higher_is_worse`)
  - short semantic description

This keeps exports classifier-agnostic while enabling future feature selection, normalization and explainability dashboards.

### Family-sensitivity analytics

The harness now computes:

- aggregate per-object-type family severity scores
- family sensitivity matrix (`LOW/MEDIUM/HIGH` + numeric scores)
- dominant and secondary failure families
- family severity ranking
- confusion-style expected-vs-observed dominant family validation
- baseline calibration from `good_ball` samples (`mean/std`)
- z-score-based feature activation summaries
- global feature discriminativeness ranking (between-type separation proxy)

## Canonical FeatureDataset Layer (Classifier-Studio Bridge)

A first-pass canonical dataset layer now formalizes semantic feature exports into a classifier-agnostic contract:

- implementation module: `vision_3d_acquisition/ml/features/dataset.py`
- core entities:
  - `FeatureSchema` (versioned feature contract + grouped semantics + metadata)
  - `FeatureSample` (single object/sample with grouped vectors + labels + provenance metadata)
  - `FeatureDataset` (dataset container, validation, stats, persistence, matrix/label extraction)

### Architectural boundary

This creates an explicit separation:

- pipelines compute semantic 25D features and analytics exports
- Classifier Studio (future) consumes canonical `FeatureDataset` objects

Studio/experiments no longer need direct knowledge of pipeline stages/artifact internals.

### Ingestion contracts

`FeatureDataset.from_advanced_validation_exports(...)` ingests:

- `advanced_25d_geometry_validation.csv`
- `advanced_25d_geometry_analysis.json`

and reconstructs:

- stable ordered feature vectors (schema-driven)
- semantic group hierarchy (`footprint_geometry`, `surface_geometry`, `sphere_consistency`, `damage_metrics`)
- synthetic metadata (`synthetic_object_type`, expected failure/reaction semantics)

### Validation and diagnostics

Dataset/schema validation reports structured warnings/errors for:

- missing required features
- unknown feature groups
- incompatible/invalid numeric fields
- missing labels

### Feature-matrix and metadata APIs

First-pass APIs include:

- `get_feature_matrix(...)`
- `get_feature_names(...)`
- `get_feature_groups(...)`
- `get_labels(...)`
- `select_feature_groups(...)`
- `exclude_features(...)`

These preserve semantic grouping while enabling classifier-ready numeric extraction.

### Statistics and normalization (lightweight)

`FeatureDataset` exposes:

- per-feature stats (mean/std/min/max/missing)
- per-group aggregate stats (feature count, aggregate variance, activation distribution)
- optional normalization helpers (`zscore`, `minmax`)

### Persistence

Lightweight JSON persistence is supported:

- `save_json(...)`
- `load_json(...)`

with schema version, metadata, samples, and split placeholders preserved.

### Split readiness

Dataset contract now includes future-ready split placeholders:

`{"train": [], "validation": [], "test": []}`

This is additive and intentionally minimal for upcoming experiment/split management passes.

## Experiment-Readiness Layer (Lightweight)

A first-pass experiment-readiness package now sits above `FeatureDataset`:

- package: `vision_3d_acquisition/ml/experiments/`
- contracts:
  - `DatasetSplitSet`
  - `ExperimentConfig`
  - `LabelTaxonomy`
- utilities:
  - deterministic split generation/persistence
  - dataset/config compatibility validation
  - lightweight baseline evaluation flow

### Canonical orchestration boundary

The intended flow is now explicit:

`FeatureDataset -> DatasetSplitSet -> ExperimentConfig -> Evaluator`

This remains classifier-agnostic and lightweight while enabling future Classifier Studio experiment workflows.

### Deterministic split strategies

Supported split strategies:

- `random`
- `stratified`
- `by_synthetic_object_type`
- `by_failure_family`

Split manifests are persisted as `split_manifest.json` with seed, strategy, and deterministic sample-id ordering.

### Compatibility validation

`validate_experiment_compatibility(...)` checks:

- dataset/config id consistency
- feature schema version compatibility
- feature-group and feature-selection validity
- normalization mode support
- label presence / dataset validation status

and emits structured diagnostics (errors/warnings + summary).

### Label taxonomy and mapping

`LabelTaxonomy` supports:

- aliases
- group remapping
- ignored labels
- binary or multiclass target modes

without mutating source datasets.

### Baseline evaluator

A lightweight baseline evaluator (`majority_label_baseline`) is implemented to validate end-to-end experiment plumbing:

- reads selected features + split manifests
- applies taxonomy mapping
- writes confusion/metrics artifacts
- persists experiment metadata under:
  - `ml/experiments/experiment_<id>/config.json`
  - `ml/experiments/experiment_<id>/split_manifest.json`
  - `ml/experiments/experiment_<id>/compatibility.json`
  - `ml/experiments/experiment_<id>/normalization_manifest.json`
  - `ml/experiments/experiment_<id>/evaluation.json`
  - `ml/experiments/experiment_<id>/feature_stats.json`

### Ablation and semantic-group experimentation

`ExperimentConfig.feature_selection` now supports:

- include/exclude groups
- include/exclude explicit features

to enable semantic-family ablations (e.g., consistency-only, no-damage, all-groups) with reproducible evaluation outputs.

## Canonical Feature Registry + Quality Analytics Layer

A feature-centric governance/analytics layer now extends `FeatureDataset`:

- registry: `vision_3d_acquisition/ml/features/registry.py`
- analytics: `vision_3d_acquisition/ml/features/analytics.py`

### FeatureRegistry responsibilities

- canonical feature metadata authority
- schema/group validation
- dataset compatibility checks
- audience-specific visibility (`operations`, `studio`, `classifier_studio`)

### Rich feature metadata contract

Feature metadata now supports:

- semantic identity: `name`, `group`, `display_name`
- interpretability: `description`, `short_description`, `tooltip`
- numeric semantics: `unit`, `higher_is_worse`, `expected_range`
- warning guidance: `recommended_warning_threshold`, `recommended_critical_threshold`
- normalization hint + visualization hints
- UX visibility and display hints

### Feature quality diagnostics

Quality report computes per-feature:

- missing/invalid ratios (null/NaN/inf coverage)
- variance and entropy proxy
- z-score + robust-z-score outlier ratios
- extreme value counts
- low-variance warnings

and per-group quality summaries.

### Stability analytics

Stability report computes:

- `feature_stability_score`
- `noise_sensitivity_score`
- `perturbation_variance`

using grouped synthetic sample behavior (cross-sample perturbation proxy) and produces stability rankings.

### Correlation and redundancy

Correlation report includes:

- pairwise correlation matrix
- high-correlation pair detection
- within-group redundancy warnings

for feature-selection and explainability workflows.

### Readiness scoring

Per-feature readiness combines:

- missingness
- invalidity
- outliers
- stability
- variance informativeness
- redundancy penalty
- metadata interpretability

Outputs:

- `readiness_score`
- `readiness_level` (`EXPERIMENTAL`, `GOOD`, `PRODUCTION_READY`)

plus group-level readiness summaries.

### Drift diagnostics (lightweight)

A baseline-snapshot comparison utility provides:

- mean shift
- stddev shift
- aggregate drift score

This is a deterministic, non-orchestrated first pass for feature drift awareness.

### Analytics exports

The feature analytics layer can export:

- `feature_quality_report.json`
- `feature_stability_report.json`
- `feature_correlation_report.json`
- `feature_readiness_report.json`
- `feature_drift_report.json`
- `feature_ux_summary.json`

### UX visibility semantics

Visibility stays audience-aware:

- Operations: grouped semantic health summaries only
- Studio: grouped diagnostics and warnings
- Classifier Studio: full feature analytics (readiness/correlation/distributions/metadata)

This preserves explainability while avoiding operator-facing metric clutter.

### Why sphere-consistency matters

Visible cap fit alone can remain deceptively good for truncated objects. The synthetic truncated-sphere mode validates that sphere-consistency/volume-deficit metrics fail even when some visible-surface fit metrics still appear acceptable, improving physical interpretability and reducing false "good ball" confidence.

## Feature UX + Runtime Data-Contract Integration

A compact UX/data-contract pass now bridges feature governance into runtime payloads without exposing heavy analytics internals.

### Canonical UX contracts

Module: `vision_3d_acquisition/ml/features/ux_contracts.py`

Defines compact, deterministic contracts:

- `FeatureGroupSummary`: group status, severity score, readiness, key evidence, top warnings
- `FeatureWarning`: canonical warning with severity/message/group and audience visibility
- `FeatureReadinessSummary`: overall readiness + per-group readiness

Severity mapping is deterministic and stable:

- `LOW`: `[0.0, 0.3)`
- `MEDIUM`: `[0.3, 0.7)`
- `HIGH`: `[0.7, 1.0]`

### Runtime payload integration

`ClassifyMiningBall25DStage` now attaches per-object compact UX fields:

- `feature_group_summaries`
- `feature_warnings`
- `feature_readiness`

These fields survive result serialization through `DetectedObject` contract additions.

### Compact audience-specific exports

Classification stage now emits compact UX artifacts:

- `feature_runtime_summary.json` (operations-facing)
- `feature_studio_summary.json` (studio/classifier-studio facing)

and publishes them via `files` references and processing artifacts.

### Visibility boundaries

Filtering utilities now enforce audience scope:

- `filter_for_operations(...)`: semantic health + actionable warnings only
- `filter_for_studio(...)`: grouped summaries, readiness, studio-visible warnings
- `filter_for_classifier_studio(...)`: grouped summaries/readiness/warnings for feature-engineering UX

No heavy correlation/stability matrices are injected into runtime object payloads.

### Operational UX contract

Operations receives compact semantic states (example mapping):

- `surface_coherence`
- `damage_evidence`
- `sphere_consistency`
- `footprint_shape`

### Contract boundary statement

- Operational UX consumes semantic summaries.
- Studio consumes grouped engineering diagnostics.
- Classifier Studio consumes full feature analytics.

This preserves explainability while preventing engineering-noise overload in runtime/operator flows.
