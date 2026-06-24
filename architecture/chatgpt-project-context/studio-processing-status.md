# Studio Processing Status Semantics

## Explicit selection model

Studio selection is normalized as:

1. Take (input data)
2. Pipeline definition (explicitly selected)
3. Optional pipeline instance (for parameterized ProcessService flows)

Execution actions are bound to selected take + selected pipeline.

## Contextual status semantics

Status filters are now family-aware:

- if selected pipeline family is `3d`:
  - unprocessed = no `DONE` marker
- if selected pipeline family is `2d`:
  - unprocessed = no completed 2D run linkage for the take

UI uses contextual wording (`Needs <family> processing`).

## Pipeline availability by modality

Pipeline options are filtered by selected take modalities using required modality matching.

Examples:

- point-cloud take: shows 3D-compatible pipelines
- RGB take: shows RGB-compatible pipelines

## Run history scope

Run history is shown for selected take + selected pipeline context.

- 2D: pipeline instance runs filtered by `summary.take_id`
- 3D: current processed result summary for selected take

## Execution actions

Buttons now call backend endpoints with explicit pipeline context:

- Run pipeline
- Reprocess
- Run until selected stage (currently routed as full run; stage-limited execution is future enhancement)
- Clear outputs

## Remaining limitations

- stage-limited 3D execution is not yet implemented; endpoint is prepared and currently runs full pipeline
- richer per-run logs/streaming progress are future work

## Explicit pipeline selection UX

Studio now follows an explicit `take -> pipeline -> run` flow:

- user selects take
- user explicitly selects pipeline definition from grouped compatible/incompatible selector
- user runs selected pipeline with explicit execution actions

### Discovery behavior

- active pipeline selector groups:
  - compatible pipelines first
  - incompatible pipelines second (muted with inline reason)
- available pipelines panel supports:
  - search by name/description/id
  - family filter (`2d`, `3d`, `generic`)
- pipeline card exposes metadata:
  - family
  - backend
  - supported modalities
  - description

### Auto-selection behavior

On take change, Studio auto-selects in this order:

1. last pipeline used for that take
2. compatible pipeline matching modality
3. preferred defaults (`3d_ball_inspection` or `mining_steel_ball_classification_2d`)
4. first compatible pipeline
5. fallback first pipeline only when none are compatible

### Execution safety

Execution buttons are disabled for incompatible pipelines and include explicit reason text/tooltips.

### Contextual trace

Execution trace card now shows selected pipeline, family, selected run/status, and timestamp in pipeline-selection context.

## Canonical selected pipeline state

`ProcessingLabPage` now uses a single canonical `selectedPipelineId` state.

The following are all derived from that single source of truth:

- Active Processing Pipeline card title
- compatibility status
- stage strip/cards
- execution actions
- execution trace block
- run history scope
- inspector pipeline context
- family-based take status semantics

## Selector placement and layout

Pipeline selector/discovery controls are rendered inside the **Active Processing Pipeline** card in the center panel, directly under the card heading.

This removes prior ambiguity where sidebar and center sections could display conflicting pipeline context.

Stage tabs remain below pipeline status and execution trace, avoiding overlap.

## UX simplification pass (compact workflow)

Studio layout is now intentionally composed as:

1. Header (take id, modality chips, run actions)
2. Compact pipeline context card
3. Stage progress cards
4. Run/trace summary
5. Result workspace tabs + content

### Compact pipeline context card

The active pipeline card was reduced to:

- one selector control (`Pipeline [name ▼]`)
- compact chips (`family`, `backend`, `compatibility`, modalities)
- one compatibility message line

Large metadata tiles (family/backend/queue/current-stage blocks) were removed from this card.

### Discovery popover behavior

Pipeline discovery is no longer always expanded inline.

- selector opens a popover with search + family filter
- list is grouped as compatible first, incompatible second
- incompatible rows remain selectable for inspection
- popover is absolutely layered and does not push page content down

### Redundancy removal rules

- top card shows pipeline metadata once
- stage strip shows stage labels/timing only
- execution trace focuses on selected run context
- inspector focuses on current stage/object/artifact details

### Take card compact status rules

Take list now uses compact family labels:

- `2D: done|not run|<status>`
- `3D: done|not run|unavailable|<status>`
- `GENERIC` is hidden when not relevant

For modality-incompatible families we show `unavailable` instead of `not run`.

### Inspector context title

Inspector title is no longer hardcoded to segmentation context.

Priority:

1. selected object
2. selected artifact
3. selected stage name (if result exists)
4. `No result selected`

## Stage-centric navigation model

Studio now uses an explicit hierarchy:

- Take
- Pipeline
- Run
- Stage
- Stage-scoped result/artifact views

Interaction semantics:

- stage strip is the primary navigator (`which processing step am I inspecting?`)
- workspace tabs are secondary and stage-contextual (`how do I inspect this stage output?`)
- selecting a tab never changes selected stage
- selecting a stage updates available views; invalid current view falls back with visible message

Execution trace is run context only and does not control stage selection.

## Stage-semantic visualization model

Studio now applies a stage capability registry that defines:

- stage category (`input`, `segmentation`, `geometry`, `measurement`, `classification`, `fusion`)
- default stage view
- stage-native view list (for example threshold mask, contour overlay, fit metrics, classification summary)
- stage-specific empty states

Workspace behavior:

- selected stage owns workspace identity
- tabs are generated from stage semantic definition (not global categories)
- renderers are selected by view renderer type (`image`, `overlay`, `table`, `metrics`, `histogram`, `json`)
- generic artifact browsing remains available as secondary debugging context

Execution trace behavior:

- default run context is compact (`run/status/timestamp/duration`)
- detailed trace graph is collapsed under on-demand expansion

## Source artifact binding (input stage usability pre-run)

Studio now resolves Input-stage views through a layered model:

1. run artifacts (preferred when present)
2. source/take artifacts (fallback)

## 2D calibration diagnostics UX

Calibration 2D detection now defaults to selected-capture processing and surfaces diagnostics for failures:

- dictionary used
- marker count + marker IDs
- ChArUco corner count
- API mode (`legacy`, `detector`, `aruco_fallback`)
- sharpness estimate
- board coverage estimate
- failure reason and warnings

Camera runtime controls are best-effort by source/backend. Unsupported controls remain visible but disabled to avoid hidden state and to preserve predictable calibration UX.

Realtime monitoring for calibration tuning now uses MJPEG stream views inside Camera Controls modal; the manager page intentionally avoids stale informational live-preview duplication.

This allows Input stage to be immediately usable before any execution run exists.

Input stage bindings:

- Image: source RGB/reflectance/heightmap asset when no run image exists
- Metadata: take/acquisition metadata
- Histogram: backend histogram endpoint (`/api/takes/{take_id}/source-histogram`) with cached/precomputed payload
- JSON: source take context (`modalities`, `assets`, `metadata`, `frameset`)

Non-input stages remain run-artifact driven and keep semantic empty states until runs produce outputs.

## Compatibility behavior in execution controls

Execution buttons use canonical selected pipeline compatibility.

- incompatible selected pipeline => run buttons disabled with explicit reason
- compatible selected pipeline => run/reprocess enabled
- run-until-selected-stage is guarded by compatibility and stage selection

## Stage-semantic threshold workspace behavior

For the selected `Threshold + morphology` stage, Studio now resolves stage-native views from segmentation artifacts:

- `Threshold mask`

## Sidebar simplification (operational-first)

Studio sidebar is intentionally reduced to operational navigation:

- dataset selector
- session selector
- search
- compact operational filters (`modality`, processing status, archived toggle)
- compact thumbnail take list with lightweight processing status chips

Removed from Studio sidebar (moved to Datasets ownership):

- category/validation/object-id semantic filters
- reference/golden curation toggles
- labeling-heavy and bulk semantic editing controls
- dataset/session curation actions

Design intent:

- reduce persistent left-rail density
- recover horizontal width for the center engineering workspace
- make stage debugging and reruns the primary visual focus

## Final lightweight refinement pass

The final pass keeps behavior unchanged while reducing cognitive load:

- batch actions are present but collapsed/de-emphasized (`Batch` disclosure)
- session labels are explicit and non-ambiguous:
  - `Experiment session` (canonical Studio replay/reprocessing scope)
  - `Runtime acquisition group` (advanced operational grouping)
- capture is secondary via collapsed treatment and lightweight affordance
- selected-take context is compact orientation (not a secondary inspector)
- Datasets handoff copy is concise and low-weight

Navigation hierarchy in Studio sidebar is now:

1. scope selectors (dataset/experiment session, with runtime grouping tucked under advanced)
2. lightweight search + operational filters
3. take browsing/selection
4. optional batch/capture disclosures

This preserves stage-centric engineering flow and avoids reintroducing semantic-curation controls.
- `Cleaned mask`
- `Overlay`
- `Morphology params`
- `JSON`

Behavior:

- stage default view is `Overlay`
- stage image views filter to the exact semantic artifact (`threshold_mask` vs `cleaned_mask`)
- params view shows threshold + morphology parameters and connected-component metrics
- inspector exposes segmentation diagnostics (components, coverage, threshold/morph params)
- empty states are stage-semantic (not generic artifact placeholders)

Threshold/morphology diagnostics now expose:

- components before/after cleanup

## Blob candidate visualization and synchronization

Blob/Contour Detection stage now presents explicit candidate visualization modes:

- `Contours`
- `Labels`
- `BBox`
- `Filled mask`
- `Rejected only`

Legend semantics:

- green: accepted candidate
- red: rejected candidate
- cyan-highlight: selected candidate

Synchronization behavior:

- overlay click -> candidate selected in table and inspector
- table row click -> candidate highlighted in overlay and inspector
- hover shows compact candidate tooltip (id, area, diameter, circularity/aspect, status)

Weak-detection guidance is now contextual:

- no candidates: suggests threshold/ROI/cleanup tuning
- all rejected: suggests min-area/circularity/border filter adjustments

## Ellipse fitting stage workspace

`ellipse_fitting` is now a stage-semantic geometry workspace with views:

- `Ellipse overlay`
- `Metrics`
- `Candidates table`
- `JSON`

The stage stays classification-agnostic and prepares downstream rules by exposing diameter and fit-quality metrics per candidate.
- removed components
- threshold/cleaned foreground coverage
- threshold mode/value/invert/blur
- morphology operation + open/close kernels + area filters
- ROI enablement and coordinates

Cleaned-mask view now warns when foreground is empty so users can tune threshold/area parameters without leaving stage context.

Visual-first threshold workspace refinement:

- image/mask/overlay render before metadata details
- compact diagnostics appear directly under visuals
- artifact metadata moved to secondary expandable details

ROI interaction refinement:

- click-drag ROI selection in stage image views
- ROI rectangle rendered across threshold/cleaned/overlay views
- apply / clear / apply + rerun actions write threshold ROI params (`roi_enabled`, `roi_x`, `roi_y`, `roi_width`, `roi_height`)

Dual ROI support:

- rectangle ROI remains fully supported
- polygon ROI added (point-by-point drawing with close on double-click/Enter, cancel on Esc, remove-last on Backspace)
- ROI mode selector (`Rectangle` / `Polygon`) controls draw behavior
- polygon coordinates persist as `roi_polygon_points` with `roi_type=polygon`

Vertical-band ROI support (conveyor/heightmap):

- ROI mode selector now includes `Vertical band ROI`
- semantics: constrain processing to X-range while preserving full scan height
- interaction model:
  - drag inside band = horizontal move
  - drag left/right edges = width resize
  - no top/bottom handles
- rendering model:
  - full-height translucent vertical band
  - auto-extends to current image height on every render
- serialization model:
  - `type=vertical_band`, `x`, `width`
  - backend keeps `type` in debug/config payloads and normalizes mask to full-height rectangle

Detection-stage semantics:

- stage-scoped geometry views are now candidate-centric (`Candidate overlay`, `Labels`, `Blob metrics`, `JSON`)
- segmentation summaries and inspector diagnostics consume structured cleanup metrics with legacy fallback
- cleaned mask is treated as candidate-generation mask for downstream blob extraction

Blob/Contour Detection workspace binding:

- stage views resolve detection artifacts with alias fallback (`blob_debug_overlay`/`candidate_overlay`/`contour_overlay`, `blob_labels`/`labels`, `blob_metrics`/`detection_metrics`, `blob_contours`/`contours`)
- summary cards and inspector use artifact-backed candidate metrics instead of only final classified objects
- candidate table supports row selection and selection state drives inspector blob details

## Calibration workspace status

Calibration navigation now exposes four tabs:

- `3D Plane` (active)
- `2D Camera` (active)
- `Laser Line` (placeholder)
- `Fusion` (placeholder)

The 2D workspace includes capture/detect/calibrate/save/set-default flow with diagnostics (reprojection, distortion, mm/px, capture count) and graceful fallback when no active 2D calibration exists.

## Experiment browser extension (Dataset-first)

Processing Lab sidebar now behaves as an experiment browser:

- dataset selection
- dataset-session filtering
- search by take id/friendly name
- tag filter
- validation status filter

Take cards now show:

- friendly name
- thumbnail preview (source-first, overlay fallback)
- tags
- modality summary
- latest run status

Take management actions are lightweight and in-context:

- rename take
- edit notes
- edit tags/labels
- set expected class
- set expected diameter (mm)
- set validation status
- rerun latest pipeline

This keeps the Studio workflow fast for iterative industrial validation without altering existing execution semantics.

2D Camera calibration now uses discovered source selection and explicit freshness labels to avoid stale preview confusion. Refresh controls are available; start/stop live controls are currently UI-disabled placeholders.

## Object-level review workflow (geometry stages)

Studio geometry workflows now include object-level review metadata editing:

- Blob/Contour and Ellipse candidate contexts expose annotation controls in Inspector.
- Candidate tables display currently assigned annotation labels.
- Annotation edits persist to dataset take sidecar metadata and survive reruns through candidate matching heuristics.

This adds a lightweight ground-truth loop for mining-ball POC validation while keeping run artifacts immutable.

2D Camera calibration now separates:

- live preview (informational)
- captured calibration frames (authoritative)

Stale preview badges remain visible for operator awareness, but stale preview no longer blocks capture/detect/calibrate flow.

## Inline experiment setup workflow

To reduce setup friction during iterative validation, Studio sidebar includes inline `+` actions beside dataset/session selectors.

This keeps experiment setup in-place:

- create dataset/session without leaving Processing Lab
- immediate auto-selection of created entity
- no full-screen modal and no navigation interruption
- lightweight pending/error feedback inline

## Studio capture-to-dataset UX

A lightweight inline capture panel is available in the Studio sidebar near dataset/session selectors.

Behavior:

- Disabled until dataset + session are selected with explicit reason text.
- Inline fields for friendly metadata before capture.
- Loading/error state is shown inline.
- Successful capture auto-selects the newly created take and refreshes the list.
- Optional quick action: `Capture + Run` to immediately process the new take.

This keeps the acquisition -> processing -> annotation loop inside one Studio workflow.

### Stable take selection during refresh

Processing Lab now separates user-driven take selection from auto-selection during background refresh:

- User clicks remain authoritative while the selected take still exists in the refreshed list.
- Auto-selection fallback only runs when no take is selected or when the selected take no longer exists.
- Polling/refetches for takes, pipelines, runtime state, and related summaries do not force-select a different take.
- Reordered take lists do not reset selection by position.

## Safe take management UX

Take Management now includes explicit lifecycle actions:

- `Remove from dataset`
- `Archive take` / `Restore archived take`
- `Delete permanently (advanced)`

### Safety semantics

- Remove action is metadata-only and preserves raw/processed data.
- Archive action hides takes from default list while preserving artifacts.
- Permanent delete is gated by typed confirmation and clearly states deletion scope.

### Archived visibility

Studio sidebar now includes a `Show archived` toggle:

- Off by default (archived hidden)
- When enabled, archived takes appear and can be restored

## 2.5D Visualization Defaults

- For `heightmap` / `derived_25d` takes, Studio now treats height preview as the primary human-facing visualization.
- Diagnostic masks remain available but are secondary.

### Priority Rules

- Heightmap takes:
  1. `heightmap_preview.png`
  2. `reflectance.png`
  3. raw upload fallback
  4. segmentation/debug masks
- RGB takes continue to prioritize RGB/overlay imagery.

### Stage Defaults

- Input stage for 2.5D defaults to `Height preview`.
- Segmentation stage for 2.5D defaults to overlay-on-height preview.
- Threshold/cleaned masks and morphology artifacts remain diagnostics, explicitly labeled as such.

### Canonical 2.5D Segmentation Overlay Artifact

- Added `height_segmentation_overlay` (`height_segmentation_overlay.png`) with:
  - `overlay_type: segmentation`
  - `coordinate_space: image_pixel`
  - `target_artifact_id: heightmap_preview`

This preserves existing artifact contracts while making human-readable inspection the default UX.
# Engineering Mode UI Semantics

Studio now supports a lightweight operator UX with an additive engineering debug mode.

## Mode Behavior

- `Operator`: compact overlays, primary object/classification summaries.
- `Engineering`: full diagnostics (residual artifacts, hover inspector, line profiles, provenance tabs).

## Hover Inspector Architecture

Hover diagnostics are implemented in projection-aware stage renderers and reused across stage views.

- Pixel `x/y`
- Approx metric value from stage range metadata when available
- Extensible fields for world coordinates, residuals, segmentation ids, normals, and reflectance

## Profile Tool

Engineering users can define a two-point line profile in stage image views to inspect sampled height/intensity evolution, distance, and min/max/p95.

## Stage-Native Engineering Tabs

Debug surfaces are integrated through semantic stage view definitions (no floating tools or detached debug windows), including tabs such as:

- `residuals`
- `diagnostics`
- `profiles`
- `provenance`

## Measurement/Height-Volume workspace semantics (stage-centric)

The Measurement stage workspace is now explicitly segmented by semantic scope to prevent layout instability and semantic overload.

### Region ownership

- Stage config:
  - top `Measurement controls` strip only
  - contains known-cube calibration controls, correction toggles, target selection, expected XYZ, and apply/reset/rerun actions
  - does not include KPIs or object rows
- Stage summary:
  - single compact KPI row directly below controls
  - metrics limited to `Objects`, `Rejected`, `Avg Ø`, `Circ.`, `Scale X`, `Scale Y`, `Scale Z`
- Object navigation:
  - left panel only (scrollable object browser/list)
- Selected object detail:
  - right panel only (selected-object summary, overlay/artifact preview, measurement table and diagnostics)
- Diagnostics:
  - remains in inspector context; not duplicated into controls/KPI regions

### Layout ownership rules

- Measurement table is no longer co-owned by KPI auto-grid layout.
- KPI cards and table are isolated in different containers.
- Table section owns full row width inside selected-object workspace (`grid-column: 1 / -1`).
- No card/table overlap is allowed under variable content lengths.

### Responsive + stability contract

- Controls are compact, fixed-purpose, and wrap horizontally on narrow widths.
- KPI cards use fixed height with single-line labels (`ellipsis`, no multiline growth).
- Left object panel and right selected-object workspace have independent scroll behavior.
- No upward flow of object detail rows under summary cards.
- Inspector remains an independent column/scroll region.

### Object-centric model (future-ready)

The Measurement stage prioritizes object inspection and validation over dashboard density. New diagnostics (residual plots, hover probes, calibration overlays, point-cloud/profile views, fusion metrics) should extend the selected-object workspace or inspector, not reintroduce floating KPI-card blocks.

### Numeric-vs-display contract

- `heightmap_preview.png` and `normalized_heightmap.png` are display artifacts only.
- Numeric measurements and hover diagnostics must use canonical numeric sources (`height16.tif`, `heightmap_frame.npz`, `normalized_heightmap.npz`) or direct derivatives serialized for numeric sampling.
- Studio must not infer physical values by inverting colorized PNG RGB pixels.


### Normalized-height semantic invariant

- Stage `Normalize heights to reference` uses canonical display semantic `height_above_plane_mm` where `0 mm` is the belt plane.
- Studio hover footer separates semantic spaces:
  - displayed semantic value (primary)
  - plane/reference value
  - optional raw diagnostics (`raw_sensor_z_mm`, residual, clipping state)
- Normalized colorbar label must explicitly indicate semantic units (`Height above plane (mm)`).
- Display transform metadata (`normalized_heightmap_display.json`) carries semantic id, raw semantic id, display range, and scaling mode.
- Viewer synchronization invariant: rendered normalized view, hover probe, legend range, and histogram all consume the same semantic raster source (`normalized_heightmap.npz`) and transform metadata.

### Second-pass hardening: canonical height semantics

- `HeightLegend` is the canonical Studio legend component for height views. It renders semantic label, units, range, direction, authoritative/debug status, and optional percentile/tick details from semantic metadata only.
- Studio now uses semantic-first artifact resolution (`semantic_field` + `representation`) for height views and numeric hover sources, with legacy fallback warnings for pre-contract artifacts.
- Runtime hover consistency checks surface debug warnings when displayed preview semantic and hover semantic differ.
- Inspector includes a dedicated "Height semantics" block showing semantic field, source semantic, lineage, units, authoritative flag, hover semantic source, numeric raster source, preview source, and color-scale range.
- Canonical geometry policy is explicit:
  - measurements/classification/reports/exports/default overlays use `height_above_belt`;
  - `raw_sensor_z` remains debug/engineering only.

### Canonical color mapping (renderer/legend/hover synchronization)

- A single shared `HeightColorMapping` contract drives the image renderer, `HeightLegend`, hover sampler and debug reconstruction.
- Resolution priority: explicit `color_mapping` block on artifact metadata → render context (`render_vmin/vmax`, `colormap_id`) → display metadata (`color_scale_min/max`, `color_map`) → legacy fallback (warning).
- `HeightLegend` now renders its gradient from the active LUT (`turbo`/`viridis`/`magma`/`gray`) instead of a hard-coded CSS gradient, eliminating the previous visual mismatch with backend-rendered previews.
- Direction is data-native (LUT anchored at `value_min`→cool, `value_max`→hot); the `direction` field only flips label/tick orientation in the legend.
- Hover/debug reconstruction uses the same named LUT for scalar↔RGB conversion, so renderer-vs-reconstruction diffing stays LUT-consistent across colormap choices.
# Studio processing status

## Dataset curation status update

Studio dataset management now includes additive curated-acquisition controls:

- session classification (`engineering|curated|benchmark|operational`)
- reference/golden take flags
- curation categories (`empty_belt_reference`, `calibration_reference`, `golden_sample`, etc.)
- sidebar filtering by session type/category/reference/golden status

These changes are metadata-only and keep processing/runtime architecture invariants intact (immutable takes, many runs per take, unchanged output layout/contracts).

## Acquisition-centric refinement (semantic/visual pass)

Studio Processing Lab now differentiates rendering intent by context without splitting apps or routes:

### Acquisition-centric browsing (default in sidebar/take browser)

- intended for dataset/session curation and acquisition review
- persistent compact **Curation context** panel exposes:
  - dataset
  - session
  - session type
  - session tags
- take cards prioritize:
  - full-frame thumbnail
  - friendly name
  - semantic chips (session type, reference/golden, categories)
- processing/classification fields remain visible but secondary
- **Selected Take Context** resolves from the currently selected take metadata/summary first (dataset/session/type/categories/reference/golden), independent of filter controls.
- filter controls remain separate browse constraints; when active, they are shown as secondary “filters currently applied” semantics.
- Selected Take Context now includes a compact active-take linkage treatment (active take chip + mini thumbnail) to visually connect sidebar context with current workspace selection.

### Classification-centric views (unchanged in runtime/inspection contexts)

- stage overlays, class labels, and diagnostics remain first-class in processing/runtime flows
- no pipeline/stage architecture changes were introduced
- Runtime/Operations cards preserve classification emphasis, while acquisition thumbnails keep full-frame `contain` rendering for better geometry readability.
- Runtime card polish improves density/balance:
  - larger full-frame thumbnails
  - reduced center watermark dominance
  - tighter horizontal composition without changing operational meaning/order.

### Toolbar hierarchy refinement

Studio top toolbar keeps the same controls and execution semantics, but visual priority is tuned as:

1. acquisition identity (`Take`)
2. processing pipeline
3. stage selection
4. compatibility + actions

This is typography/spacing/chip-weight refinement only (no workflow or route changes).

### Inspector density refinement

- Inspector sections remain complete diagnostically.
- Vertical spacing/padding is compacted for better information density and faster scanability.

## Studio loading model (summary-first)

Studio loading is now explicitly split for responsiveness:

- Sidebar listing uses paginated `TakeSummary` pages (`/api/takes/paged`) with server-side filters.
- Selected take detail (`/api/takes/{take_id}`) loads independently and lazily.
- Heavy artifact/result hydration remains selected-take scoped (stage workspace + inspector), not list-scoped.

### Pagination semantics

- `limit` + `offset`, newest-first ordering.
- response includes `items`, `has_more`, `next_offset`.
- filters remain equivalent to prior `/api/takes` semantics.

### Frontend safeguards

- filter text inputs are debounced.
- stale in-flight list responses are ignored.
- pagination resets on filter changes.
- selected take remains stable when possible via resolver logic.

## Operations card composition (runtime)

Runtime monitoring rows now use balanced 3-column semantics:

1. status/acquisition metadata
2. classification summary (label, confidence, object count, superclass)
3. full-frame preview (`contain`, letterboxed)

Operational emphasis remains classification-centric while preserving acquisition readability in previews.

### Semantic chip system

Compact chips are now the primary semantic language for:

- session type: `engineering`, `curated`, `benchmark`, `operational`
- take semantics: `reference`, `golden`, category/reference tags

### Thumbnail strategy

Acquisition browsing thumbnails use full-frame preview semantics:

- `object-fit: contain`
- aspect ratio preserved
- letterboxing accepted
- no aggressive object-centric crop for browse cards

### Acquisition identity hierarchy

Stable acquisition identity is explicitly surfaced as:

1. dataset/session context
2. friendly take identity + semantic chips
3. immutable `take_id` and processing metadata

This keeps acquisition semantics stable across pipeline/rule/model reruns while preserving many-runs-per-take processing behavior.

## Dataset Entity Drawer Integration Boundary

- Added Dataset Session drawer action `Open in Studio filtered to session`.
- This preserves Studio ownership of execution/runtime inspection while allowing Datasets users to navigate with session context.
- No processing execution/status logic was moved into Datasets drawer.
- Drawer surfaces only semantic summary and metadata-management context.
