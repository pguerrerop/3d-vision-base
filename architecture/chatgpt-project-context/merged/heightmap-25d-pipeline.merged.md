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

All major height artifacts include transform lineage metadata:

- `semantic_field`
- `derived_from`
- `transform` (`type`, and optional transform details like `plane_id`)

### Numeric raster contract

The pipeline persists canonical semantic numeric rasters:

- `raw_sensor_z.values.f32`
- `plane_signed_distance.values.f32`
- `height_above_belt.values.f32`

Each semantic raster carries metadata:

- `semantic_field`
- `coordinate_space`
- `units`
- `dtype`
- `source_stage`

### Semantic-first artifact model

Consumers resolve artifacts by `(semantic_field, representation)` instead of filename. Legacy filename compatibility remains additive.

### Hover/render consistency guarantee

Hover sampling and numeric diagnostics consume semantic rasters only. Colorized previews remain visualization-only.

### Canonical color mapping contract (`HeightColorMapping`)

A single canonical contract is shared by every height visualization consumer (image renderer, `HeightLegend`, hover sampler, debug reconstruction) so the colorbar can never visually disagree with the rendered preview:

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

- LUT is anchored data-natively (`value_min` → cool, `value_max` → hot); `direction` only changes legend labels/ticks.
- backend persists `color_mapping` on the rendered preview artifact, the display metadata JSON, the render context artifact and the canonical numeric raster artifact.
- frontend resolution priority: explicit `color_mapping` → render context → display metadata → legacy fallback (warning).
- `HeightLegend` draws the gradient from the same named LUT used by the renderer (`turbo`/`viridis`/`magma`/`gray`); no hard-coded gradients.
- hover/debug reconstruction uses the same `colorMap` + `colorScaleMin/Max` for scalar↔RGB transforms.

## Advanced Geometric Ball-Quality Semantics (v2 additive)

The `measurement` + `classification` stages expose physically-grounded grouped metrics while preserving legacy flat fields.

### Semantic metric groups

Per-object payloads include:

- `footprint_geometry`
- `surface_geometry`
- `sphere_consistency`
- `damage_metrics`

Groups are propagated to `classification_explanation.metrics_used` and `semantic_group_summaries`.

### Coordinate/space assumptions

- `footprint_geometry`: XY contour/mask geometry in calibrated metric space.
- `surface_geometry` and `damage_metrics`: visible 2.5D cap points from `height_above_belt`.
- `sphere_consistency`: visible-cap vs inferred sphere-cap coherence (occluded hemisphere is inferred).

### Metric definitions

- Circularity: `C = 4*pi*A / P^2`.
- Radial boundary uniformity: radial mean/std/CV/max deviation from contour centroid.
- Sphere fit: least-squares sphere on visible XYZ points (center/radius/RMSE/max).
- Ellipsoid proxy fit: PCA axes/aspect/residual proxy.
- Sphere-vs-ellipsoid gain: normalized residual improvement.
- Radial height residual: visible heights vs expected spherical cap.
- Surface completeness + volume coherence: occupancy/volume proxy vs expected sphere volume.
- Damage proxies: roughness, flat-region area ratio, discontinuity/high-curvature ratios.

### Classification and explainability

- Heuristics consume grouped evidence without collapsing to a single monolithic score.
- Explanations include per-group pass/fail summaries and per-metric contribution rules across footprint/surface/consistency/damage families.

### Studio exposure

- Inspector renders compact grouped summaries (Footprint, Surface, Sphere consistency, Damage).
- Classification rule view renders semantic group summaries before detailed rule rows.

### Limits

- Ellipsoid fit is currently PCA-proxy based.
- Visible-surface-only geometry limits certainty for hidden-side damage.
- Damage proxies are geometric; RGB crack cues are future fusion inputs.

### Future readiness

Grouped metrics are modular and contract-safe for RGB+25D fusion, ML export, and dataset analytics.

## Synthetic Geometry Validation Harness

Additive validation framework for metric-family discrimination under controlled defect geometry.

### Generator

- `vision_3d_acquisition/synthetic/advanced_geometry_25d.py`
- Object types:
  - `good_ball`
  - `worn_ellipsoid`
  - `truncated_sphere`
  - `chipped_sphere`
  - `flattened_ball`
  - `elongated_scrap`

### Realism controls

- gaussian height noise
- missing/no-return regions
- belt tilt
- edge aliasing perturbation

Runs are deterministic via explicit seeds.

### Metadata contract

Synthetic take metadata persists:

- `synthetic_object_type`
- `generation_parameters`
- `expected_failure_modes`
- `expected_metric_family_reactions`
- acquisition-effects config

Debug GT assets: `synthetic_gt_mask.png`, `synthetic_flat_regions_gt.png`.

### Benchmark runner

- `scripts/run_advanced_25d_geometry_validation.py`
- Outputs:
  - `data/reports/advanced_25d_metric_family_report.json`
  - `data/reports/advanced_25d_metric_family_report.md`
  - `data/reports/advanced_25d_geometry_validation.csv`
  - `data/reports/advanced_25d_geometry_analysis.json`

### Classifier-studio readiness

Aggregate analytics export includes:

- `feature_schema_version: "v1"`
- grouped `feature_groups` (`footprint_geometry`, `surface_geometry`, `sphere_consistency`, `damage_metrics`)
- lightweight `feature_metadata` (group, unit, `higher_is_worse`, semantic description)

This supports future classifier-studio ingestion and explainability-first feature dashboards without binding to a specific model.

### Aggregate sensitivity analytics

The harness now computes:

- per-object-type family score aggregates
- family sensitivity matrix (`LOW/MEDIUM/HIGH` + numeric severities)
- dominant/secondary failure families and ranking
- expected-vs-observed dominant-family confusion summaries
- `good_ball` baseline calibration stats (mean/std)
- z-score feature activation summaries
- global feature discriminativeness ranking.

## Canonical FeatureDataset Layer

First-pass canonical dataset abstraction now exists in:

- `vision_3d_acquisition/ml/features/dataset.py`

Core contracts:

- `FeatureSchema`: versioned grouped feature definitions + metadata
- `FeatureSample`: per-object semantic feature sample
- `FeatureDataset`: dataset container with validation/stats/persistence/matrix extraction

### Classifier-studio alignment

Formal bridge is now:

`25D semantic feature exports -> FeatureDataset -> Classifier Studio / experiments`

This keeps classifier consumers decoupled from pipeline-stage/artifact internals.

### Ingestion

`FeatureDataset.from_advanced_validation_exports(...)` ingests:

- `advanced_25d_geometry_validation.csv`
- `advanced_25d_geometry_analysis.json`

and reconstructs stable ordered vectors with preserved semantic groups.

### Validation/statistics/utilities

Added first-pass capabilities:

- structured schema/sample validation (missing/unknown/invalid diagnostics)
- per-feature and per-group statistics
- feature-matrix and label extraction APIs
- semantic group selection/exclusion utilities
- lightweight normalization helpers (`zscore`, `minmax`)
- JSON persistence/reload with schema-version + metadata preservation
- split placeholders (`train`/`validation`/`test`) for future workflows.

### Physical rationale

Truncated-sphere cases demonstrate why visible-surface fit alone is insufficient: surface fit may remain partly acceptable while sphere-consistency and volume-deficit metrics fail, which is the desired physically grounded behavior.

## Experiment-Readiness Layer

Lightweight experiment layer now exists at:

- `vision_3d_acquisition/ml/experiments/`

Core contracts:

- `DatasetSplitSet`
- `ExperimentConfig`
- `LabelTaxonomy`

### Canonical boundary

`FeatureDataset -> SplitSet -> ExperimentConfig -> Evaluator`

This is the intended Classifier Studio integration surface and remains backend-agnostic.

### Deterministic splits

Strategies supported:

- `random`
- `stratified`
- `by_synthetic_object_type`
- `by_failure_family`

Split manifests are persisted and reloadable (`split_manifest.json`) with deterministic ordering and seed linkage.

### Compatibility checks

Experiment compatibility validator checks schema versions, feature/group availability, normalization mode support, and label availability, returning structured diagnostics.

### Label taxonomy

Taxonomy utilities support aliases, group remapping, ignored labels, and binary-vs-multiclass mapping without mutating source datasets.

### Baseline evaluation flow

A lightweight `majority_label_baseline` evaluator is included to validate end-to-end experiment plumbing and artifact generation:

- config + split + compatibility + normalization manifest
- confusion/evaluation metrics
- feature statistics

Artifact layout:

- `ml/experiments/experiment_<id>/config.json`
- `ml/experiments/experiment_<id>/split_manifest.json`
- `ml/experiments/experiment_<id>/compatibility.json`
- `ml/experiments/experiment_<id>/normalization_manifest.json`
- `ml/experiments/experiment_<id>/evaluation.json`
- `ml/experiments/experiment_<id>/feature_stats.json`

### Semantic ablations

`ExperimentConfig.feature_selection` supports include/exclude groups/features for semantic-family ablation studies while preserving grouped feature semantics.

## Feature Registry + Quality/Stability Analytics

Feature-governance and feature-analytics layer now exists:

- `vision_3d_acquisition/ml/features/registry.py`
- `vision_3d_acquisition/ml/features/analytics.py`

### FeatureRegistry

Acts as canonical source for:

- feature metadata
- semantic grouping
- schema/dataset compatibility checks
- audience-specific visibility rules (`operations`, `studio`, `classifier_studio`)

### Rich metadata semantics

Feature metadata now includes:

- display/explainability fields (`display_name`, `short_description`, `tooltip`)
- numeric semantics (`unit`, `higher_is_worse`, `expected_range`)
- warning guidance thresholds
- normalization/visualization hints
- UX visibility flags and display precision hints

### Quality analytics

Per-feature diagnostics include:

- missing/invalid ratios
- variance + entropy proxy
- z-score and robust-z-score outlier detection
- extreme-value and low-variance warnings

Plus per-group quality summaries.

### Stability analytics

Adds:

- `feature_stability_score`
- `noise_sensitivity_score`
- `perturbation_variance`

with feature and group rankings.

### Correlation and redundancy

Exports correlation matrix, strongest high-correlation pairs, and redundancy warnings (especially within semantic families).

### Readiness scoring

Readiness combines missingness, validity, outliers, stability, variance informativeness, redundancy penalty, and metadata interpretability.

Readiness levels:

- `EXPERIMENTAL`
- `GOOD`
- `PRODUCTION_READY`

### Drift diagnostics

Lightweight baseline-snapshot comparisons provide mean/stddev shifts and drift scores (first-pass, deterministic, non-orchestrated).

### Export outputs

Analytics export files:

- `feature_quality_report.json`
- `feature_stability_report.json`
- `feature_correlation_report.json`
- `feature_readiness_report.json`
- `feature_drift_report.json`
- `feature_ux_summary.json`

### UX audience semantics

- Operations: semantic group health summaries only
- Studio: grouped diagnostics + warnings
- Classifier Studio: full feature analytics and metadata

## Feature UX + Runtime Data-Contract Integration

A compact UX/data-contract layer was added on top of feature governance to make runtime and studio payloads operationally consumable.

### Canonical UX contracts

Implemented in `vision_3d_acquisition/ml/features/ux_contracts.py`:

- `FeatureGroupSummary`
- `FeatureWarning`
- `FeatureReadinessSummary`

plus deterministic severity mapping (`LOW`, `MEDIUM`, `HIGH`) and audience filters.

### Runtime/studio payload additions

Per-object classification payload now carries:

- `feature_group_summaries`
- `feature_warnings`
- `feature_readiness`

Result contracts were extended so these fields persist through serialization.

### Compact exports

Classification stage now writes:

- `feature_runtime_summary.json`
- `feature_studio_summary.json`

and links both through `files` and processing artifacts.

### Visibility policy

- Operations: compact semantic statuses + actionable warnings only.
- Studio: grouped summary cards + readiness + warning evidence.
- Classifier Studio: richer feature-engineering-oriented summaries (with future link to full analytics).

Runtime payloads intentionally avoid dumping full correlation/stability/analytics matrices.

### Architectural boundary

Operational UX consumes semantic summaries.
Studio consumes grouped engineering diagnostics.
Classifier Studio consumes full feature analytics.
