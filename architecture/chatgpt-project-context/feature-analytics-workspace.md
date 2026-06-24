# Feature Analytics Workspace (Phase 1)

## Rationale
Feature Analytics is now a first-class Studio workspace for dataset/session/run/stage-aware feature exploration, separability debugging, and explainability.

## Architecture
- Dedicated navigation entry: `Studio -> Feature Analytics` (`/feature-analytics`)
- Additive implementation, reusing existing:
  - dataset/session metadata
  - stage-semantic object outputs
  - object annotation linkage
  - pipeline/run metadata

## Contracts
### FeatureRecord
- `object_id`, `take_id`, `dataset_id`, `session_id`
- `pipeline_id`, `run_id`, `stage_id`
- `labels`, `superclass`
- `features: Record<feature_key, number>`
- `acquisition_metadata`, `calibration_metadata`
- `timestamp`, `validation_status`, `split`

### FeatureDefinition
- `feature_key`, `display_name`
- `semantic_group` (geometry/height/morphology/classification/runtime)
- `unit`, `description`
- `source_stage`, `source_metric_path`

### FeatureAnalyticsQuery
Supported filters:
- `datasets`, `sessions`, `labels`, `superclass`
- `validation_status`, `split`
- `pipeline`, `calibration`
- `date_from`, `date_to`
- `feature_selection`

Session filter semantics:
- Phase 1 UI presents a single selected session scoped by the active dataset.
- Session is treated as an acquisition/experiment context rather than a free-text token.
- Query contract remains comparison-friendly so future multi-session comparison can extend without reworking histogram semantics.

## Backend Endpoints
- `GET /api/feature-analytics/features`
  - Returns normalized feature definitions for filtered records.
- `GET /api/feature-analytics/distributions`
  - Histogram/distribution by `feature_key`.
  - `group_by=superclass|label`, `bins`, `mode=count|density`.
- `GET /api/feature-analytics/scatter`
  - Phase-1 scaffold for future scatter/pair workflows.
- `GET /api/feature-analytics/objects`
  - Linked inspection payload for selected feature ranges.

Backend performs filtering + bucketing server-side to keep frontend lightweight.

## UI Semantics (Phase 1)
- Left sidebar filters
- Top controls: feature/grouping/mode/bin count
- Main visualization: grouped histogram overlays
- Right inspector: feature metadata + stats
- Linked samples table: click bin range -> load object/take rows -> open take detail

### Session Selection Semantics
- Session selector is dataset-scoped and behaves like a first-class context selector rather than a text field.
- Session options expose compact metadata in dropdown labels:
  - session type
  - created date
  - take count
- This reinforces sessions as acquisition/runtime/calibration conditions and prepares later comparison workflows.

### Histogram Axis Semantics
- X-axis now includes adaptive numeric tick labels derived from the active feature range.
- Tick density is adaptive (readable by default) rather than labeling every bin.
- Optional toggle `Show bin edge labels` overlays exact bin boundary labels.
- Numeric labels use compact formatting to avoid floating-point noise.

### Bin Tooltip Behavior
- Bin hover tooltip includes:
  - grouping label/class
  - bin min/max values
  - count
  - density signal (explicit in normalized mode, derived from group count in count mode)
- Selected bin range remains visible above linked samples for context persistence.

### Comparative Distribution Interpretation
- The main chart should be read as a comparative distribution-band view, not only as a classical histogram.
- Left-side band labels align each row to its grouped class/superclass for rapid scanning.
- This emphasizes class-wise distribution shape and tail behavior while preserving histogram bin interaction.

### UNKNOWN Visual Semantics
- `UNKNOWN` is intentionally muted relative to authoritative classes.
- Lower opacity and lighter/dashed treatment communicate uncertainty without removing it from analysis.
- Goal is to distinguish unresolved distribution mass from trusted labeled bands.

### Linked Sample Scrolling
- Linked samples moved to an internal scroll container with sticky header.
- Histogram remains visible while browsing large linked sample sets.
- Table panel no longer expands page height unboundedly.

### Object-Level Thumbnail Semantics
- Linked sample rows now include a compact object preview column.
- Thumbnail resolution follows object-level linkage semantics first:
  1. `object_crop_{object_id}` artifact when available
  2. source-image crop using object bbox/candidate geometry
  3. take-level thumbnail fallback
  4. neutral placeholder fallback
- Crop behavior is square, centered, and padded around bbox to reduce contour clipping.

### Hover Preview Behavior
- Hovering the compact thumbnail opens a non-blocking larger preview near cursor.
- Hover card includes quick identifiers (`take_id`, `object_id`, `superclass`, feature value) for rapid visual triage.
- UX is tuned for high-density scanning, not gallery browsing.

### Performance Rationale
- Thumbnails are served via a lightweight cached endpoint:
  - `GET /api/takes/{take_id}/objects/{object_id}/thumbnail`
- Endpoint creates/stores small cached JPEG thumbs in take stage cache.
- Frontend uses lazy image loading and fixed-size previews to avoid expensive full-frame rendering.
- Layout remains stable if preview is missing or unavailable.

### Responsive Workspace Composition
- Main analytics column balances histogram and linked samples with bounded min/max heights.
- Target composition is roughly split between chart and table regions for analysis continuity.
- Mobile/tablet collapse preserves internal scrolling behavior and compact controls.

### Filter Evolution Rationale
- Filter panel is lightly sectioned into basic and advanced groups to keep density high while reducing form sprawl.
- This is a preparatory step, not a dashboard redesign.
- Grouping helps future additions like calibration/session comparison without introducing filter-builder complexity.

## Linkage Model
Link uses stable `take_id + object_id` pairs, enriched with `dataset/session/pipeline/run/stage` context and optional matched object annotation payload.

## Renderer Decisions
- Keep Phase 1 lightweight with existing Studio page patterns and client API layer.
- Use explicit controls and compact engineering-focused layout.
- Keep chart contract generic for future KDE/violin/scatter/UMAP/PCA.

## Limitations (Phase 1)
- Histogram only (no KDE/violin/pair yet)
- Scatter endpoint is scaffold-level
- Feature source extraction currently centered on object metrics in `result.objects` + `rejected_objects`
- Very large datasets still rely on synchronous aggregation path

## Future Work
- Add async aggregation cache/index
- Add run/stage comparison mode
- Add calibration/split differential views
- Add multi-session comparison mode using the existing session-as-context semantics
- Add embedding feature groups and dimensional projections
- Add drift monitoring and model explainability overlays
- Add linked-sample row virtualization for very large selections (prepared by fixed-height scrolling container)
- Add visual-cluster exploration mode (feature bin -> thumbnail strip -> fast outlier triage) while keeping compact analytical density

## Physical Object Alignment
- Feature Analytics now has backend grouping support for `physical_object_id` in addition to superclass/label.
- This prepares repeated-observation analysis where intra-object variability can be compared against inter-object variability.
- Physical Objects remain semantic grouping entities; detected object feature extraction remains unchanged.

## Feature UX Runtime Contract
- Compact feature UX contracts now bridge semantic feature governance into runtime and Studio payloads.
- Canonical object-level fields are:
  - `feature_group_summaries`
  - `feature_warnings`
  - `feature_readiness`
- These fields are intentionally compact and serialization-safe so they can survive result persistence, API transport, and Studio rendering.

### Audience Boundaries
- Operations consumes compact semantic statuses and actionable warnings only.
- Studio consumes grouped engineering diagnostics with compact evidence and readiness badges.
- Classifier Studio consumes richer feature-engineering analytics and metadata.

### Runtime Artifact Shape
- 25D classification now emits:
  - `feature_runtime_summary.json`
  - `feature_studio_summary.json`
- Classification explanation payloads and result summaries also carry compact feature readiness/runtime summary blocks, avoiding the need for UI layers to inspect raw analytics dumps.

### Explainability Intent
- Semantic feature groups remain first-class:
  - `footprint_geometry`
  - `surface_geometry`
  - `sphere_consistency`
  - `damage_metrics`
- UX surfaces should summarize these groups as health/evidence contracts rather than flattening them into engineering-heavy metric tables for operational audiences.
