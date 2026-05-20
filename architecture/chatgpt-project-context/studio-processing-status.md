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
