# 2.5D Mining-Ball Current Pipeline Spec

Status: current-behavior read-only spec for `mining_steel_ball_classification_25d`

Purpose: document the pipeline exactly as it behaves today before any algorithm changes.

Primary implementation sources:

- `vision_3d_acquisition/apps/ball_inspection_25d/pipeline.py`
- `vision_3d_acquisition/vision_core/pipelines/stages_25d.py`
- `vision_3d_acquisition/pipelines/registry.py`
- `frontend/src/components/stageSemantics.ts`
- `frontend/src/components/stage_view_renderers/index.tsx`

This document is intentionally descriptive, not normative. If code and older docs disagree, this file follows the code.

## 1. Pipeline Overview

### 1A. Public / Studio stages

The public pipeline registry exposes the following stage chain for `mining_steel_ball_classification_25d`:

| Order | Stage id | Human label | Inputs | Outputs | Always? | Source |
|---|---|---|---|---|---|---|
| 1 | `input` | Load heightmap capture | `heightmap`, optional `reflectance`/`rgb` | decoded capture artifacts | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 2 | `detect_belt_plane` | Detect reference surface | decoded heightmap | reference-surface model, masks, residual diagnostics, stripe diagnostics | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 3 | `normalize_heights_to_plane` | Normalize heights to reference | raw heightmap + selected reference model | `height_above_belt` rasters, normalized preview, diagnostics | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 4 | `remove_belt_segment_objects` | Remove reference + segment objects | normalized height + reference/stripe masks | threshold masks, cleaned object mask, segmentation overlay | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 5 | `geometry` | Footprint geometry | segmentation mask | connected components, contour / hull / ellipse geometry | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 6 | `measurement` | Height + volume metrics | geometry + normalized height | per-object height / volume / deformation metrics | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 7 | `measurement_diagnostics` | Measurement diagnostics | measurement outputs | feature vector, provenance, quality flags | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 8 | `classification` | Mining-ball classification | measurement diagnostics | class labels, explanations, classifier outputs | yes | `vision_3d_acquisition/pipelines/registry.py` |
| 9 | `overlay` | Overlay rendering | processed objects and stage artifacts | human-facing overlays | yes | `vision_3d_acquisition/pipelines/registry.py` |

### 1B. Internal execution stages / classes

The actual runner executes a more granular internal sequence:

| Order | Internal class | Public / Studio stage | Function |
|---|---|---|---|
| 1 | `LoadCaptureStage` | `input` | resolve take, output dir, metadata, source files |
| 2 | `LoadHeightmapCaptureStage` | `input` | decode heightmap into `HeightmapFrame` |
| 3 | `ApplyCalibration25DStage` | `input` / calibration pre-step | attach 2.5D calibration context |
| 4 | `DetectBeltPlaneStage` | `detect_belt_plane` | reference-surface selection, plane fit, stripe filter, residual diagnostics |
| 5 | `NormalizeHeightsToPlaneStage` | `normalize_heights_to_plane` | create canonical `height_above_belt` field |
| 6 | `RemoveBeltAndSegmentObjectsStage` | `remove_belt_segment_objects` | threshold and suppress reference/stripe pixels |
| 7 | `ExtractConnectedComponentsStage` | `geometry` | label components, create object candidates |
| 8 | `FitObjectGeometryStage` | `geometry` | contour/hull/ellipse/object geometry |
| 9 | `ComputeHeightMetricsStage` | `measurement` | height stats, volume proxy, grouped metrics |
| 10 | `ValidateKnownObjectScale25DStage` | no dedicated public stage | optional known-object scale validation from metadata |
| 11 | `ComputeMeasurementDiagnosticsStage` | `measurement_diagnostics` | quality, features, provenance, flags |
| 12 | `ClassifyMiningBall25DStage` | `classification` | heuristic / rule-based classification |
| 13 | `Generate25DOverlaysStage` | `overlay` | overlay rendering and summary artifacts |
| 14 | `SerializeProcessingResultStage` | result serialization | final `result.json` payload |

Implementation: `vision_3d_acquisition/apps/ball_inspection_25d/pipeline.py`

### 1C. Persisted artifacts and result files

Persisted outputs are the union of:

- explicit `files[...]` entries maintained by each stage
- explicit `context.add_processing_artifact(...)` calls
- final `result.json`

Key persisted families:

- Detect/reference artifacts:
  `belt_plane.json`, `raw_heightmap_preview.png`, `valid_mask.png`, `plane_fit_roi_mask.png`, `background_candidate_mask.png`, `background_seed_mask.png`, `expanded_plane_mask.png`, `final_plane_inlier_mask.png`, `plane_inlier_mask.png`, `depth_gradient_magnitude.png`, `low_gradient_mask.png`, `reference_surface_selected_mask.png`, `reference_surface_candidates.json`, `reference_surface_plateaus.json`, `flat_candidate_histogram.json`, `background_depth_plot.png`, `background_plateau_plot.png`, `flat_candidate_depth_plot.png`, `plane_residual_histogram.json`, `plane_fit_debug.json`, `selected_surface_debug.json`, `background_selection_debug.json`, `gradient_debug.json`
- Stripe artifacts:
  `belt_stripes_mask.png`, `belt_base_mask.png`, `belt_stripes_tophat_mask.png`, `belt_stripes_shape_mask.png`, `belt_above_belt_mask.png`, `belt_wide_object_mask.png`, `belt_baseline_local_min.png`, `belt_altitude_local_min.png`, `belt_altitude_histogram.png`, `belt_altitude_histogram.json`, `belt_stripe_filter_debug.json`
- Normalization artifacts:
  `normalized_heightmap.npz`, `normalized_heightmap.png`, `height_above_belt.values.f32`, `raw_sensor_z.values.f32`, `plane_signed_distance.values.f32`, `normalized_heightmap_display.json`, `normalized_heightmap.render_context.json`, `normalized_height_histogram.json`, `normalization_debug.json`
- Segmentation artifacts:
  `foreground_before_plane_suppression.png`, `below_reference_mask.png`, `above_threshold_mask.png`, `plane_suppressed_mask.png`, `rejected_background_residuals.png`, `rejected_belt_stripes.png`, `normalized_height_threshold_mask.png`, `cleaned_object_mask.png`, `final_object_mask.png`, `segmentation_overlay.png`, `connected_components_overlay.png`, `segmentation_debug.json`

### Stage-by-stage summary

| Stage id | Label | Implementation | Main inputs | Main outputs | Downstream consumers | Conditional? |
|---|---|---|---|---|---|---|
| `input` | Load heightmap capture | `LoadCaptureStage`, `LoadHeightmapCaptureStage`, `ApplyCalibration25DStage` | take files, metadata | `heightmap_frame`, metadata, calibration context | all later stages | always |
| `detect_belt_plane` | Detect reference surface | `DetectBeltPlaneStage` | `heightmap_frame`, ROI config, stage params | `belt_plane`, `reference_surface_model`, plane/stripe masks, residual map, debug JSONs | normalization, segmentation, Studio | always |
| `normalize_heights_to_plane` | Normalize heights to reference | `NormalizeHeightsToPlaneStage` | `heightmap_frame`, `belt_plane`, `reference_surface_model`, `plane_inlier_mask` | `normalized_heightmap_mm`, `height_above_belt_raster`, preview PNG, histogram/debug JSON | segmentation, geometry, measurement, classification | always |
| `remove_belt_segment_objects` | Remove reference + segment objects | `RemoveBeltAndSegmentObjectsStage` | normalized height, `reference_surface_selected_mask` fallback chain, `belt_stripes_mask_array` | segmentation mask, threshold/cleaned masks, overlay, segmentation debug | geometry | always |
| `geometry` | Footprint geometry | `ExtractConnectedComponentsStage`, `FitObjectGeometryStage` | segmentation mask, normalized height | labeled objects, geometry summaries | measurement | always |
| `measurement` | Height + volume metrics | `ComputeHeightMetricsStage` | objects + canonical height raster | grouped physical metrics | diagnostics, classification | always |
| `measurement_diagnostics` | Measurement diagnostics | `ValidateKnownObjectScale25DStage`, `ComputeMeasurementDiagnosticsStage` | measurement outputs, metadata | quality flags, feature vectors, optional known-object validation | classification, Studio | known-object validation is optional, diagnostics stage always runs |
| `classification` | Mining-ball classification | `ClassifyMiningBall25DStage` | features + object metrics | class labels, explanations, classification artifacts | overlay, result serialization | always |
| `overlay` | Overlay rendering | `Generate25DOverlaysStage` | objects + stage artifacts | human-facing overlays | result serialization, Studio | always |

## 2. Belt / Reference-Surface Process

### High-level inputs

Primary input field:

- `frame.z_mm` from `HeightmapFrame`

Supporting masks and metadata:

- `frame.valid_mask`
- stage ROI from `plane_fit_roi` or metadata-derived ROI
- optional tuning parameters from `stage_params.detect_belt_plane`

Important representation note:

- belt/reference selection runs on raw sensor-space Z, not on `height_above_belt`
- `height_above_belt` does not exist until `NormalizeHeightsToPlaneStage`

### ROI semantics

ROI is used only for reference-surface estimation, not for final normalization or object cropping.

Current behavior:

1. Resolve ROI from `self.plane_fit_roi` or metadata via `_roi_from_metadata(...)`.
2. Normalize region mode to one of `none`, `full_height_x_band`, `rectangle`, `polygon`.
3. Build `roi_mask`.
4. Compute `valid_for_fit = frame.valid_mask & roi_mask`.

Persisted diagnostic:

- `plane_fit_roi_mask.png`

### Candidate-selection strategies

The stage supports three conceptual paths:

1. Percentile-based background selection using `background_selection_mode`
2. `low_gradient_surface`
3. `low_gradient_depth_plateaus`

Current default code path is the low-gradient family when `background_detection_strategy` is one of:

- `low_gradient_surface`
- `low_gradient_depth_plateaus`

The architecture note in `architecture/chatgpt-project-context/heightmap-25d-pipeline.md` is incomplete here because the current implementation contains both low-gradient strategies plus the older percentile path.

### Common precomputation

When enough valid ROI pixels exist:

1. Compute `near_q` and `far_q` from raw Z percentiles.
2. Build `low_mask` and `high_mask`.
3. Run `_reference_height_gate_from_distribution(...)`.
4. Use `height_gate_mask` only for low-gradient strategies; otherwise keep `valid_for_fit`.

Height gate purpose:

- clip foreground-dominated height modes away from reference selection
- reduce accidental selection of tall but flat surfaces

Relevant debug JSON field:

- `height_gate` inside `gradient_debug.json`

### `low_gradient_surface` strategy

Current flow:

1. Compute depth gradient magnitude by `_compute_depth_gradient(...)`.
2. Threshold it by `fixed`, `otsu`, or percentile mode.
3. Build `low_grad_mask`.
4. Optionally apply open + close morphology to the low-gradient mask.
5. Use `_select_low_gradient_component(...)` to score connected components.

Component scoring uses:

- area ratio
- Z constancy via `z_std` and `z_mad`
- gradient compactness via `gradient_p95`
- border touch bonus
- depth preference term

The selected component is then narrowed again:

1. Compute median / MAD of selected Z values.
2. Build a support window in Z.
3. Reject high-gradient ridge pixels inside the component.
4. Keep only the filtered component as `candidate_mask`.
5. Store removed pixels as `raised_candidate_rejected_mask`.

Artifacts:

- `depth_gradient_magnitude.png`
- `low_gradient_mask.png`
- `low_gradient_components_overlay.png`
- `low_gradient_components.json`
- `reference_surface_candidates.json`
- `gradient_debug.json`

### `low_gradient_depth_plateaus` strategy

This is the richer conveyor-belt-specific path.

Current flow:

1. Build `flat_candidates_pre_hessian = low_grad_mask & height_gate_mask`.
2. Optionally refine by low-Hessian filtering.
3. Write `flat_candidate_mask.png`.
4. Collect Z values from flat candidates.
5. Run `_detect_depth_plateaus(...)` to detect flat Z plateaus in the filtered Z histogram.
6. Run `_select_background_plateau(...)` to pick the belt plateau.
7. Restrict `candidate_mask` to the selected Z band.
8. Mark higher flat plateaus as `raised_candidate_rejected_mask`.

Plateau selection modes:

- `lowest_dominant`
- `largest`
- `lowest`
- `robust_band`

Default current semantics:

- prefer the lowest plateau whose area fraction exceeds `low_gradient_plateau_select_min_area_fraction`
- if none qualify, try robust median ± MAD band
- otherwise fall back to largest plateau

Fallback states:

- no flat candidates
- no plateau detected, so all flat candidates become `candidate_mask`
- robust-band fallback selected

Artifacts:

- `reference_surface_plateaus.json`
- `reference_surface_candidates.json`
- `flat_candidate_histogram.json`
- `background_plateau_plot.png`
- `flat_candidate_depth_plot.png`
- `background_selected_plateau_mask.png`
- `rejected_raised_plateau_mask.png`
- `gradient_debug.json`

### Belt-stripe filtering inside `DetectBeltPlaneStage`

Stripe handling happens during `detect_belt_plane`, before final reference support is frozen.

It runs only in the `low_gradient_depth_plateaus` strategy branch.

Current effect:

1. takes the selected background plateau as raw belt-support candidate
2. removes stripe-like raised texture from that support
3. writes stripe/base diagnostic masks
4. updates `candidate_mask`
5. subtracts stripes again from expanded plane masks and final plane inliers
6. exports a stripe suppression mask for downstream segmentation

See section 3 for full detail.

### Component / border filtering after candidate selection

After a candidate mask is chosen, generic component filtering may still run:

1. connected components on `candidate_mask`
2. reject small components below `background_candidate_min_component_area`
3. optionally require border touch using `background_must_touch_roi_border`

Special automatic fallback:

- if border-touch filtering removes all candidates and `background_selection_mode == "automatic"`, the code may flip the inferred depth convention and retry the opposite percentile side

Warnings produced here include:

- `border_connected_filtering_removed_all_candidates`
- `automatic_convention_flipped_due_to_border_prior`
- `selected_reference_surface_too_small`
- `low_gradient_mask_coverage_too_low`
- `background_candidate_selection_too_sparse`

### Plane fit / refit

Once `candidate_mask` exists:

1. Convert selected pixels to XYZ via `_heightmap_points(...)`.
2. Downsample if above `plane_fit_downsample`.
3. Compute candidate Z MAD.
4. Set `effective_residual_threshold_mm` as the max of:
   - `plane_fit_residual_threshold_mm`
   - `plane_fit_residual_threshold_adaptive_multiplier * cand_z_mad`
5. Fit plane by `_fit_plane_ransac(...)`.
6. Measure sampled inlier ratio.
7. Mark failure if inlier ratio is below `plane_fit_min_inlier_ratio`.

### Residual expansion logic

If a plane was fit:

1. compute residual map over the full ROI-valid area
2. choose residual tolerance:
   - fixed `plane_background_residual_tolerance_mm`, or
   - adaptive using seed residual std and `plane_background_residual_adaptive_multiplier`
3. expand to `expanded_plane_mask = abs(residual) <= tol`
4. optionally close / flood-fill holes
5. subtract `stripes_mask` if present
6. record `expanded_plane_coverage`

If enabled, refit:

1. convert expanded inliers back to points
2. least-squares fit plane by `_least_squares_plane(...)`
3. recompute residuals
4. recompute final inlier mask under the same tolerance
5. subtract `stripes_mask` again

Final inlier semantics:

- if `final_plane_inlier_mask` exists, `belt_mask = final_plane_inlier_mask`
- otherwise fallback to `abs(residual_map) <= effective_residual_threshold_mm`
- then subtract stripes again from `belt_mask`

### Fallback behavior

Deterministic fallback plane:

- if no plane coefficients exist, create horizontal plane at median valid Z

Reference model selection:

- `reference_surface_model == "plane"`: force plane
- `reference_surface_model == "constant_z"`: force constant-Z
- `reference_surface_model == "auto"`: use constant-Z fallback when plane fit is degraded

Auto-fallback triggers include:

- stage status not `success`
- low inlier ratio
- high residual p95

When constant-Z is used:

- `reference_surface_model_type = "constant_z"`
- `coeffs = [0, 0, 1, -constant_z_mm]`
- residual map becomes `raw_z_mm - constant_z_mm`

### Exact mask semantics

Current meaning of the main masks:

| Mask / field | Meaning |
|---|---|
| `background_seed_mask` | initial selected reference-support mask after candidate selection and later filtering; stored from `seed_mask = candidate_mask.copy()` before residual expansion |
| `background_candidate_mask` | persisted PNG of the final selected candidate reference surface (`candidate_mask`) |
| `reference_surface_selected_mask` | same conceptual support mask as `candidate_mask`; Studio “Selected surface” prefers this |
| `expanded_plane_mask` | ROI-valid pixels within residual tolerance of the fitted plane, after optional fill/close and stripe subtraction |
| `final_plane_inlier_mask` | refit version of expanded plane mask if refit happened; later exported as `plane_inlier_mask` / `belt_mask` |
| `plane_inlier_mask` | final authoritative inlier mask used by later diagnostics and normalization background probe |
| `belt_plane_residuals` | absolute residual heatmap PNG for visualization |
| `plane_signed_distance.values.f32` | signed residual raster |
| `normalized_heightmap` | canonical `height_above_belt` preview, produced later in normalization stage |

### Exact parameters and defaults

The detect/reference stage exposes a large parameter surface. The stripe/segmentation tables later in this document list the most important ones; the canonical defaults live in `DetectBeltPlaneStage` and `RemoveBeltAndSegmentObjectsStage`.

### Exact warnings / diagnostics

Observed warnings emitted by the detect stage include:

- `plateau_strategy_no_flat_candidates`
- `plateau_strategy_no_plateau_detected`
- `plateau_strategy_used_robust_band_fallback`
- `belt_stripe_filter_removed_large_bg_fraction`
- `border_connected_filtering_removed_all_candidates`
- `automatic_convention_flipped_due_to_border_prior`
- `selected_reference_surface_too_small`
- `low_gradient_mask_coverage_too_low`
- `background_selection_bias_toward_zero_depth`
- `background_candidate_selection_too_sparse`
- `expanded_plane_coverage_too_low`
- `background_not_near_zero_after_plane_fit`
- `foreground_background_separation_too_small`
- `inlier_ratio_very_low`
- `plane_fit_used_very_few_samples`
- `extreme_plane_fit_residuals_detected`
- `model_residual_p95_too_high`
- `background_candidates_poorly_overlap_plane_inliers`
- `constant_z_fallback_used`
- ROI coverage warnings added into `background_selection_debug`

Primary diagnostics files:

- `gradient_debug.json`
- `background_selection_debug.json`
- `selected_surface_debug.json`
- `plane_fit_debug.json`
- `belt_plane.json`
- `plane_residual_histogram.json`
- `belt_stripe_filter_debug.json`

## 3. Belt-Stripe Detection Process

This section describes the current stripe logic exactly as implemented by `_compute_belt_stripe_filter(...)`.

### Where stripe detection happens

Current placement:

- stage: `DetectBeltPlaneStage`
- strategy branch: only `low_gradient_depth_plateaus`
- input domain: raw `z_mm`
- timing: before final plane support is frozen, before normalization, before segmentation

Therefore stripe detection does **not** operate on `height_above_belt`. It operates on raw Z over a chosen domain, with the selected plateau used as a reference only for the shape pass.

### Input representation

Inputs to `_compute_belt_stripe_filter(...)`:

- `z_mm`
- `domain_mask`
- `pixel_size_mm`
- `bg_plateau_mask`
- stripe-related thresholds and morphology settings

The caller currently passes:

- `domain_mask = valid_for_fit` when `belt_stripe_filter_scope == "global"`
- `domain_mask = candidate_mask` when scope is `bg_plateau`
- `bg_plateau_mask = candidate_mask` before candidate narrowing

Important validity rule:

- non-finite and non-positive Z are explicitly removed from the morphology domain before any stripe processing

### What “belt base” and “stripes” mean

Within the stripe helper:

- `belt_base_mask = domain minus accepted stripe pixels`
- `stripes_mask = accepted stripe pixels`

These are not object masks. They are reference-support diagnostics and suppression masks.

### Pass 1: top-hat / local-min pass

The first pass is local-baseline morphology.

Current logic:

1. Choose kernel width from `window_mm / pixel_size_mm`.
2. Compute a baseline and altitude map by `_hat(...)`.
3. If `direction == "raised"`, use:
   - `baseline = opening(z)` when `baseline_mode == "opening"`
   - else `baseline = erosion(z)`
   - `altitude = z - baseline`
4. If `direction == "recessed"`, use the symmetric closing / local-max formulation.
5. If `direction == "auto"`, compute both raised and recessed variants and choose the one with larger bimodality score.

Bimodality score:

- `p95(altitude) - p50(altitude)`

Thresholding:

- `fixed`: `fixed_threshold_mm`
- `k_mad`: `median + k_mad * MAD`
- `otsu`: `_otsu_threshold(...)`
- final threshold is clamped to at least `min_altitude_mm`

Top-hat stripe candidates:

- `candidate_stripes_tophat = domain_mask & (altitude_map > threshold)`

Artifacts from this pass:

- `belt_baseline_local_min.png`
- `belt_altitude_local_min.png`
- `belt_altitude_histogram.png`
- `belt_altitude_histogram.json`
- `belt_stripes_tophat_mask.png`

### What “local-min baseline” actually is

The Studio label “Local-min baseline” is slightly broader than the strict math.

Actual current behavior:

- when `baseline_mode == "opening"` and direction is raised, the baseline is a true morphological opening, not just a local minimum
- when `baseline_mode == "erosion"`, it is effectively local-min-only

So the UI label is approximately right, but not exact for the default `opening` mode.

### Pass 2: shape / z-floor pass

This complementary pass exists because the top-hat alone misses:

1. stripes wider than the morphology kernel
2. stripes adjacent to tall objects where object height contaminates the opened baseline

Current flow:

1. Estimate belt Z from `bg_plateau_mask` when possible:
   - `belt_z_estimate = median(bg_plateau_z)`
   - `belt_z_upper = percentile(bg_plateau_z, z_floor_upper_percentile)`
2. If no plateau is available, fallback to lower / upper percentiles of the whole domain.
3. For raised stripes only:
   - `reference_z = belt_z_upper` by default, or `belt_z_estimate` when `z_floor_use_upper_bound == False`
   - `threshold_low_mm = reference_z + z_floor_margin_mm`
   - `above_belt_mask = domain & (z > threshold_low_mm)`
   - optionally cap it by `threshold_high_mm = reference_z + max_stripe_height_mm`
4. Morphologically close `above_belt_mask` using `above_belt_close_mm`.
5. Morphologically open it with an object-sized kernel using `object_kernel_mm`.
6. Surviving large blobs become `wide_object_mask`.
7. The remainder becomes `shape_stripes_mask = above_belt_mask & ~wide_object_mask`.

Artifacts from this pass:

- `belt_stripes_shape_mask.png`
- `belt_above_belt_mask.png`
- `belt_wide_object_mask.png`

### Acceptance / rejection of stripe candidates

Final stripe candidates:

- `candidate_stripes = candidate_stripes_tophat | extra_stripes_shape`

Acceptance gate:

- if `stripes_fraction < min_stripe_fraction`, then the filter is considered not applied and all stripe masks are reset to empty
- otherwise stripes are accepted

Final stripe/base masks returned:

- `stripes_mask = candidate_stripes`
- `belt_base_mask = domain & ~stripes_mask`

### How stripe masks modify reference fitting

Current detect-stage effects:

1. If the stripe filter applies, `candidate_mask` is replaced by `belt_base_mask` intersected with the original background plateau.
2. `raised_candidate_rejected_mask` is merged with `stripes_mask`.
3. `expanded_plane_mask` has stripes subtracted.
4. `final_plane_inlier_mask` has stripes subtracted again after refit.
5. final `belt_mask` for residual/background stats also has stripes subtracted.

In plain terms:

- stripes shrink the plane-fit support before fit quality is frozen
- stripes are not just a visualization artifact

### How stripe masks modify downstream segmentation

The detect stage exports:

- `context["belt_stripes_mask_array"]`
- `context["belt_base_mask_array"]`

The segmentation stage then:

1. thresholds foreground from normalized height
2. suppresses selected reference-surface pixels
3. suppresses stripe pixels

This means stripe suppression affects:

- foreground threshold mask after suppression
- connected component count
- all downstream object geometry, measurements, diagnostics, and classification

### Current persisted stripe diagnostics

| Artifact id | Meaning |
|---|---|
| `belt_stripes_mask` | accepted final stripe mask |
| `belt_base_mask` | final base-support mask after stripe removal |
| `belt_stripes_tophat_mask` | top-hat-only detections |
| `belt_stripes_shape_mask` | shape/z-floor-only detections |
| `belt_above_belt_mask` | z-floor thresholded “raised” domain |
| `belt_wide_object_mask` | opened large objects retained by shape pass |
| `belt_baseline_local_min` | baseline map used for altitude computation |
| `belt_altitude_local_min` | altitude over baseline |
| `belt_altitude_histogram` | histogram of altitude values |
| `belt_stripe_filter_debug` | JSON counters, thresholds, skip reasons, chosen direction |

### Current failure modes

Likely current failure modes from code structure:

| Failure mode | Why it can happen now |
|---|---|
| real objects classified as stripes | object narrower than `object_kernel_mm`, object tails lost by opening, or object falls inside the stripe domain before segmentation |
| stripes classified as objects | stripe wider than top-hat kernel and not removed by shape pass, or z-floor disabled / too conservative |
| stripes remove too much belt support | large `stripes_fraction_of_bg`, aggressive thresholding, or curved belt upper tail treated as raised structure |
| stripe suppression interferes with plane fit | stripe mask shrinks `candidate_mask`, `expanded_plane_mask`, and final inliers |
| plane failure causes stripe artifacts | fallback `constant_z` can distort later normalization; stripe pass still depends on raw-domain assumptions |
| belt curvature causes false positives | upper-tail belt pixels can exceed `belt_z_estimate + margin`; code partly counters this by preferring `belt_z_upper` |
| object-adjacent stripes missed by top-hat | explicitly noted in code comments; shape pass is the mitigation |
| recessed stripes unsupported in shape pass | current shape pass only supports `direction_used == "raised"` |

## 4. Segmentation After Belt / Stripe Suppression

Implementation: `RemoveBeltAndSegmentObjectsStage`

### Current flow

1. Load canonical normalized height map `normalized_heightmap_mm`.
2. Compute:
   - `below_or_on_reference = normalized <= reference_tolerance_mm`
   - `foreground = normalized > min_height_mm`
   - optional `foreground &= normalized <= max_height_mm`
   - `foreground &= valid_mask`
3. Persist pre-suppression masks.
4. Load plane/reference mask using this fallback chain:
   - `reference_surface_selected_mask`
   - else `expanded_plane_mask`
   - else `plane_inlier_mask`
5. If `suppress_plane_mask_in_segmentation`, subtract that mask from foreground.
6. Load `belt_stripes_mask_array` and subtract it from foreground.
7. Persist:
   - `plane_suppressed_mask`
   - `rejected_background_residuals`
   - `rejected_belt_stripes`
8. Convert to binary threshold mask.
9. Apply morphology:
   - open
   - close
   - optional hole fill
   - optional Gaussian smoothing and re-threshold
10. Run connected components.
11. Remove components smaller than `min_component_area`.
12. Persist cleaned/final mask and overlay.

### Height-threshold semantics

- authoritative numeric field: `height_above_belt`
- `min_height_mm` is the main object foreground threshold
- `max_height_mm` is an optional upper clip
- `reference_tolerance_mm` affects the below-reference diagnostic mask, not the main threshold

### Morphology and component filtering

Current binary cleanup:

- open with square kernel `morphology_kernel`
- close with same kernel
- optional flood-fill hole filling
- optional Gaussian blur + threshold with `smoothing_kernel`
- component area filter by `min_component_area`

### Wide-object handling

There is no separate wide-object handling stage in segmentation itself.

Current meaning:

- “wide-object mask” belongs to the stripe shape pass, inside `detect_belt_plane`
- segmentation simply consumes the final stripe-suppressed foreground

### Interaction with belt/base/stripe masks

Current interactions:

- segmentation uses `reference_surface_selected_mask` rather than final plane inliers as the first suppression choice
- stripe mask is always subtracted when present
- `belt_base_mask` itself is not used directly by segmentation; it is primarily detect-stage support / diagnostics

### Artifacts shown in Studio

Primary segmentation-stage artifacts:

- `normalized_height_threshold_mask`
- `cleaned_object_mask`
- `connected_components_overlay`
- `height_segmentation`
- `segmentation_debug`

### Rejected candidates representation

Rejected areas are represented as masks, not object rows, at this stage:

- `rejected_background_residuals.png`
- `rejected_belt_stripes.png`

Segmentation debug also reports counts before and after suppression.

## 5. Artifact Dictionary

### Detect / reference / stripe / normalization / segmentation views

| Studio tab / view | Artifact id or source | File | Stage | Type | Semantic field | Measurement-authoritative? | Visual interpretation | Common failure signs |
|---|---|---|---|---|---|---|---|---|
| Raw heightmap | `raw_heightmap_preview` | `raw_heightmap_preview.png` | `detect_belt_plane` | image | `raw_sensor_z` | no | raw Z preview | broken decoding, clipped range, dead regions |
| Valid mask | `valid_mask` | `valid_mask.png` | `detect_belt_plane` | image | validity | no | valid vs no-return pixels | holes, large invalid strips |
| Depth gradient | `depth_gradient_magnitude` | `depth_gradient_magnitude.png` | `detect_belt_plane` | image | gradient magnitude | no | low values are flat surfaces | gradient everywhere high, noisy belt |
| Low-gradient mask | `low_gradient_mask` | `low_gradient_mask.png` | `detect_belt_plane` | image | low-gradient candidate mask | no | binary flat-surface candidates | too sparse or merges object + belt |
| Surface candidates | `reference_surface_candidates`, `reference_surface_plateaus`, `flat_candidate_histogram`, `gradient_debug`, `selected_surface_debug` | JSON files | `detect_belt_plane` | json | candidate / plateau metadata | no | why a surface was selected | missing plateau, suspicious candidate score |
| Selected surface | `reference_surface_selected_mask` preferred, fallback `expanded_plane_mask` | `reference_surface_selected_mask.png` | `detect_belt_plane` | image | selected reference support | no | selected belt-support pixels before final plane inliers | tiny support, obvious object leakage |
| Plane inliers | `final_plane_inlier_mask` or `plane_inlier_mask` | `final_plane_inlier_mask.png`, `plane_inlier_mask.png` | `detect_belt_plane` | image | final plane support | no | final inlier region after expansion/refit | ragged or over-large inlier region |
| Residual heatmap | `belt_plane_residuals` | `belt_plane_residuals.png` | `detect_belt_plane` | image | `plane_signed_distance` visualized as absolute residual | no | distance from fitted reference surface | bright belt, large structured residual bands |
| Residual histogram | `plane_residual_histogram` | `plane_residual_histogram.json` | `detect_belt_plane` | json/table | residual stats | no | residual distribution | heavy tails, high p95 |
| Diagnostics | `plane_fit_debug`, `selected_surface_debug`, `plane_residual_histogram` | JSON files | `detect_belt_plane` | json/table | fit/debug metadata | no | overall detect-stage health | low inlier ratio, fallback used |
| Depth plot | `background_depth_plot` | `background_depth_plot.png` | `detect_belt_plane` | image | raw depth distribution | no | percentile-side distribution | ambiguous near/far separation |
| Plateau plot | `background_plateau_plot` | `background_plateau_plot.png` | `detect_belt_plane` | image | flat-candidate Z histogram | no | selected plateau vs rejected plateaus | no clear dominant plateau |
| Filtered depth plot | `flat_candidate_depth_plot` | `flat_candidate_depth_plot.png` | `detect_belt_plane` | image | filtered flat-candidate depth sequence | no | only low-gradient candidate Z values | wide noisy band, no plateau |
| Belt base | `belt_base_mask` | `belt_base_mask.png` | `detect_belt_plane` | image | stripe-suppressed belt support | no | support pixels retained after stripe removal | too little support remains |
| Belt stripes | `belt_stripes_mask` | `belt_stripes_mask.png` | `detect_belt_plane` | image | final stripe ignore mask | no | accepted stripe pixels | large object regions included |
| Stripes — top-hat | `belt_stripes_tophat_mask` | `belt_stripes_tophat_mask.png` | `detect_belt_plane` | image | top-hat stripe candidates | no | stripes found by local-baseline altitude | misses wide stripes |
| Stripes — shape pass | `belt_stripes_shape_mask` | `belt_stripes_shape_mask.png` | `detect_belt_plane` | image | z-floor + object-width stripe candidates | no | stripes found by above-belt minus wide-object carve | over-flags elevated belt curvature |
| Above belt mask | `belt_above_belt_mask` | `belt_above_belt_mask.png` | `detect_belt_plane` | image | `z > reference_z + margin` | no | all pixels considered raised before width filtering | large belt regions included |
| Wide-object mask | `belt_wide_object_mask` | `belt_wide_object_mask.png` | `detect_belt_plane` | image | large above-belt structures | no | object-sized regions preserved by opening | objects too small disappear |
| Stripe altitude | image view prefers `belt_altitude_histogram_image` and may also match `belt_altitude_local_min` | `belt_altitude_histogram.png`, `belt_altitude_local_min.png` | `detect_belt_plane` | image | altitude over baseline | no | either histogram plot or altitude map depending on artifact selection | bimodality weak or threshold poor |
| Local-min baseline | `belt_baseline_local_min` | `belt_baseline_local_min.png` | `detect_belt_plane` | image | stripe baseline map | no | local baseline used for altitude | baseline leaks object heights |
| Normalized height | `normalized_heightmap` | `normalized_heightmap.png` | `normalize_heights_to_plane` | image | `height_above_belt` | yes | canonical physical height field | belt not near zero, sign inverted |
| Below/equal reference | `below_reference_mask` | `below_reference_mask.png` | `normalize_heights_to_plane` and segmentation alias | image | below-reference mask | no | pixels at or below reference | large positive background drift |
| Above threshold | `above_threshold_mask` | `above_threshold_mask.png` | `normalize_heights_to_plane` and segmentation alias | image | threshold mask | no | pixels above object threshold | foreground too dense or too sparse |
| Threshold mask | `normalized_height_threshold_mask` | `normalized_height_threshold_mask.png` | `remove_belt_segment_objects` | image | thresholded object candidates | no | post-suppression foreground | obvious stripe bands remain |
| Cleaned mask | `cleaned_object_mask` / `final_object_mask` | `cleaned_object_mask.png`, `final_object_mask.png` | `remove_belt_segment_objects` | image | cleaned object mask | no | post-morphology final binary objects | objects fragmented or erased |
| Connected components | `connected_components_overlay` plus component table | `connected_components_overlay.png` | `remove_belt_segment_objects` / `geometry` | overlay/table | object components | yes downstream | what survives into geometry | too many tiny components |

## 6. Parameter Dictionary

This table focuses on parameters affecting reference selection, stripes, and segmentation.

| Parameter | Default | Stage | Used by | Meaning | Effect when increased | Effect when decreased | Typical failure symptom |
|---|---|---|---|---|---|---|---|
| `gradient_threshold_percentile` | `70.0` | detect | low-gradient strategies | percentile used to define “flat enough” pixels | more permissive flat set | stricter flat set | merges object + belt, or no flat candidates |
| `low_gradient_open_kernel` | `3` | detect | low-gradient mask cleanup | opening size on low-gradient mask | removes small noise but may erase thin belt support | preserves thin structures | noisy mask or over-pruned mask |
| `low_gradient_close_kernel` | `5` | detect | low-gradient mask cleanup | closing size on low-gradient mask | bridges nearby flat regions | keeps regions separate | belt-object bridging |
| `low_gradient_min_component_area` | `1500` | detect | low-gradient surface selection | minimum low-gradient area | rejects small patches | accepts more small patches | no candidate or spurious small candidate |
| `low_gradient_plateau_hist_bins` | `96` | detect | plateau detection | histogram bins over flat-candidate Z | finer plateau separation | coarser plateau separation | unstable plateau edges or merged plateaus |
| `low_gradient_plateau_min_fraction` | `0.05` | detect | plateau detection | min fraction for plateau significance | stricter plateau acceptance | more plateau candidates | no plateau vs too many plateaus |
| `low_gradient_plateau_min_pixels` | `500` | detect | plateau detection | min pixels for plateau significance | stricter plateau acceptance | more permissive | no plateau on small ROIs |
| `low_gradient_plateau_select_min_area_fraction` | `0.20` | detect | `_select_background_plateau` | dominant-area threshold | favors dominant broad plateau | may pick smaller low-Z plateau | chevrons win or belt ignored |
| `reference_surface_min_area_ratio` | `0.08` | detect | candidate quality warning / low-gradient selection | minimum believable support fraction | expects broader support | allows tiny supports | tiny selected surface |
| `plane_fit_residual_threshold_mm` | `5.0` in code family, adaptively maxed | detect | RANSAC inlier tolerance floor | base plane residual tolerance | looser fit acceptance | stricter fit acceptance | spurious tilted plane or plane fit failure |
| `plane_fit_min_inlier_ratio` | `0.3` in current class defaults | detect | fit acceptance | minimum inlier ratio | easier fit acceptance when lowered | stricter when raised | constant-Z fallback or unstable plane |
| `plane_background_residual_tolerance_mm` | `10.0` in detect stage family | detect | expansion | residual tolerance for plane expansion | wider expanded plane | tighter expanded plane | background leak or too little support |
| `plane_background_residual_tolerance_mode` | `adaptive` | detect | expansion | fixed vs adaptive residual expansion | adaptive tolerates sensor noise | fixed can be brittle | under/over-expanded plane |
| `plane_refit_after_expansion` | `True` | detect | final inlier refinement | refit plane after expansion | usually smoother final inliers | no refit | residual bias remains |
| `reference_surface_model` | `auto` | detect / normalize | plane vs constant-Z fallback | choose reference representation | force plane if set to `plane` | force fallback if `constant_z` | degraded normalization or wrong plane |
| `belt_stripe_filter_enabled` | `True` | detect | stripe helper | enable stripe suppression | more suppression when on | no stripe removal when off | stripes become foreground |
| `belt_stripe_filter_window_mm` | `30.0` | detect | top-hat pass | morphology window for local baseline | preserves wider structures, may miss stripes | catches narrower stripes, may hit object edges | stripes missed or objects flagged |
| `belt_stripe_filter_direction` | `auto` | detect | top-hat pass | raised vs recessed logic | fixed direction reduces ambiguity | wrong direction misses stripes | shape pass skip or poor bimodality choice |
| `belt_stripe_filter_threshold_mode` | `otsu` | detect | top-hat pass | thresholding mode for altitude | depends on mode | depends on mode | under/over-detection |
| `belt_stripe_filter_min_altitude_mm` | `10.0` | detect | top-hat pass | floor on stripe altitude threshold | fewer stripes accepted | more subtle stripes accepted | stripe false negatives or belt micro-texture |
| `belt_stripe_filter_min_stripe_fraction` | `0.02` | detect | acceptance gate | minimum stripe fraction required to apply filter | filter skips more often when raised | filter applies on weak evidence when lowered | either no stripe suppression or over-eager suppression |
| `belt_stripe_filter_scope` | `global` | detect | stripe helper domain | global valid ROI vs only selected plateau | global can catch off-plateau ribs | plateau-only is safer but narrower | stripes outside plateau leak back in |
| `belt_stripe_filter_baseline_mode` | `opening` | detect | stripe helper | true opening vs erosion baseline | opening protects object edges | erosion is more aggressive | object edges flagged as stripes |
| `belt_stripe_filter_z_floor_enabled` | `True` | detect | shape pass | enable above-belt width-based stripe catch | catches wide/object-adjacent stripes | leaves only top-hat | wide stripe false negatives |
| `belt_stripe_filter_z_floor_margin_mm` | `20.0` | detect | shape pass | how far above belt Z to start “raised” class | fewer above-belt candidates | more above-belt candidates | misses stripes or flags belt curvature |
| `belt_stripe_filter_max_stripe_height_mm` | `500.0` | detect | shape pass | upper cap for stripe candidates | allows taller stripe-like structures | rejects tall raised regions | tall guides / objects pollute stripes |
| `belt_stripe_filter_above_belt_close_mm` | `30.0` | detect | shape pass | hole-bridging close before object opening | preserves wide objects better | less bridging | object tails leak into stripes |
| `belt_stripe_filter_object_kernel_mm` | `100.0` | detect | shape pass | width threshold separating objects from stripes | more regions count as stripes if raised too much? actually larger kernel means only very wide objects survive | smaller kernel lets more regions count as objects | objects become stripes or stripes become objects |
| `min_height_mm` | `8.0` | segmentation | foreground threshold | minimum `height_above_belt` to count as object | fewer candidate pixels | more candidate pixels | misses objects or leaks belt texture |
| `max_height_mm` | `None` | segmentation | foreground threshold | optional upper cutoff | removes tall outliers | keeps all tall structures | object truncation if too low |
| `suppress_plane_mask_in_segmentation` | `True` | segmentation | plane suppression | subtract reference-surface support from foreground | stronger background suppression when on | more background leaks when off | belt residuals become objects |
| `morphology_kernel` | `5` | segmentation | open/close | binary cleanup size | stronger cleanup, may erode small objects | more detail/noise retained | fragmented or noisy masks |
| `min_component_area` | `120` | segmentation | component filter | minimum connected-component area | remove more small blobs | keep more small blobs | objects disappear or many speckles remain |
| `smoothing_kernel` | `3` | segmentation | optional Gaussian cleanup | smooth binary mask before final threshold | merges nearby blobs | sharper mask | fragmentation or blur-induced merging |

Note on exact ranges:

- the code does not always enforce explicit user-facing ranges
- where no hard range is enforced, interpret “expected range” as operationally reasonable rather than validated input schema

## 7. Current Known Documentation Mismatches

| Source A | Source B | Mismatch | Assessment |
|---|---|---|---|
| `architecture/chatgpt-project-context/heightmap-25d-pipeline.md` | actual runner `apps/ball_inspection_25d/pipeline.py` | architecture note lists 8 coarse stages; runner executes 14 internal classes | stale/incomplete high-level doc |
| pipeline registry `vision_3d_acquisition/pipelines/registry.py` | actual runner | registry exposes 9 public stages, but internal stages like `ApplyCalibration25DStage`, `ExtractConnectedComponentsStage`, `ValidateKnownObjectScale25DStage`, `SerializeProcessingResultStage` are hidden | intentional public abstraction, but incomplete as execution spec |
| `docs/processing.md` | current 2.5D pipeline | still describes point-cloud requirements and older stage names such as `DecodeHeightmapStage`, `PreprocessPointCloudStage`, `PlaneFilterStage` for the “ball inspection flow” | stale |
| architecture note stage tab list | frontend `stageSemantics.ts` | architecture note lists only a minimal subset of detect-stage tabs; frontend includes plateau and stripe-specific tabs | stale/incomplete |
| Studio label “Local-min baseline” | stripe helper implementation | default baseline is morphological opening, not pure local-min erosion | label is approximate, not exact |
| Studio “Stripe altitude” image selector | artifact set | renderer may match `belt_altitude_histogram_image` or `belt_altitude_local_min`; the view label implies one concept but artifact selection spans both histogram and altitude-map naming | slightly ambiguous UI mapping |
| “Selected surface” tab | renderer mapping | frontend may show `reference_surface_selected_mask` or fallback `expanded_plane_mask` | intentional fallback, but can hide conceptual difference |
| `background_candidate_mask` naming | actual semantics | persisted image is the final selected candidate surface, not just a loose seed/candidate superset | naming is legacy/inexact |
| `plane_inlier_mask` vs `final_plane_inlier_mask` | context export | both exist; final `plane_inlier_mask` artifact is actually the post-stripe final belt mask | intentional compatibility duplication but easy to misread |
| `below_reference_mask` / `above_threshold_mask` | both normalization and segmentation stages | same artifact ids / filenames are emitted across stages for related but context-dependent uses | intentional aliasing, but can be confusing without stage context |

## 8. Current Behavior Summary

### Current truth

- The pipeline assumes the belt is a broad, relatively flat, reference-support surface visible inside a fit ROI.
- It does not directly “know” the belt; it estimates a reference surface from raw Z using low-gradient selection, plateau analysis, and plane fitting with fallback to constant Z.
- Stripes are handled inside `detect_belt_plane`, before normalization and before segmentation.
- Stripe suppression changes the actual support used for plane fitting and also the foreground that later reaches connected components.
- The final authoritative `height_above_belt` is produced in `NormalizeHeightsToPlaneStage` from either:
  - signed distance to the selected plane, or
  - raw Z relative to fallback constant Z
- Downstream segmentation, measurement, diagnostics, and classification consume `height_above_belt`, not raw Z previews.

### Most fragile current assumptions

- the belt occupies enough low-gradient area to dominate or at least form a recoverable plateau
- border-touch and area priors remain valid across ROI choices
- stripe texture is mostly raised, not recessed
- the z-floor shape pass can separate narrow stripes from genuine objects by width
- stripe suppression improves plane support more than it removes legitimate support
- constant-Z fallback is acceptable when plane fitting degrades

## Unresolved Ambiguities / Needs Confirmation

These are not code errors; they are places where current semantics are still easy to misread:

1. Studio `belt_altitude_plot` can resolve to either the histogram image or the altitude-map artifact depending on artifact ordering; the renderer allows both.
2. The segmentation stage prefers `reference_surface_selected_mask` over `expanded_plane_mask` / `plane_inlier_mask` for suppression. This is current code behavior, but if the product intent was “suppress final plane inliers,” that intent is not what the code currently does.
3. The old `background_candidate_mask` name suggests a looser seed set than what is actually persisted in current low-gradient paths.
4. Some parameter defaults are explicit in the dataclass, but no single user-facing schema documents their operational ranges; this spec records current code defaults rather than validated UI ranges.

## Files Inspected

- `vision_3d_acquisition/apps/ball_inspection_25d/pipeline.py`
- `vision_3d_acquisition/vision_core/pipelines/stages_25d.py`
- `vision_3d_acquisition/pipelines/registry.py`
- `frontend/src/components/stageSemantics.ts`
- `frontend/src/components/stage_view_renderers/index.tsx`
- `architecture/chatgpt-project-context/heightmap-25d-pipeline.md`
- `docs/processing.md`
