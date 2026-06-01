# Merged: Feature Analytics Workspace (Phase 1)

## Summary
A dedicated Feature Analytics workspace was reintroduced as a first-class Studio surface (`/feature-analytics`) with normalized contracts, backend aggregations, and linked object inspection.

## Included Decisions
- Dedicated workspace, not classifier-embedded
- Query contract spanning dataset/session/labels/superclass/split/pipeline/calibration/date
- Server-side histogram aggregation (count+density) with superclass/label grouping
- Linked bin-range sample inspection using `take_id/object_id`
- Feature metadata inspector with source-stage semantics

## API Surface
- `GET /api/feature-analytics/features`
- `GET /api/feature-analytics/distributions`
- `GET /api/feature-analytics/scatter` (phase-1 scaffold)
- `GET /api/feature-analytics/objects`

## Frontend Surface
- New product nav item: `Feature Analytics`
- Page layout:
  - left filters
  - top feature controls
  - histogram overlays
  - right metadata inspector
  - linked samples table

## UX Refinement Additions
- Histogram x-axis now exposes adaptive numeric ticks for feature-value orientation.
- Optional `Show bin edge labels` toggle adds exact boundary overlays without default clutter.
- Bin tooltips include grouping label, bin min/max, count, and density context.
- Linked samples panel uses fixed-height internal scrolling with sticky table headers.
- Workspace composition balances histogram and linked samples with bounded responsive heights.
- Current table architecture is ready for future row virtualization if linked selections grow.

## Thumbnail Linkage Additions
- Linked samples table now includes compact object-level thumbnails (`Preview` column).
- Fallback hierarchy:
  1. object crop artifact
  2. bbox-based source image crop
  3. take thumbnail
  4. neutral placeholder
- Hovering a thumbnail shows a larger non-blocking preview card with key identifiers.
- Thumbnail serving is cache-backed and lazy-loaded to preserve responsiveness and avoid image-heavy UX drift.
- Design intent remains analytical: feature-space to visual-space correlation for outlier/mislabel/segmentation drift inspection.

## Compatibility
Additive only; no existing Studio flow was removed or replaced.

## Known Gaps
- No PCA/UMAP/scatter matrix visual yet
- No cached pre-aggregation index yet
- No multimodal embedding extraction yet (contract designed to allow it)
- Linked samples are not virtualized yet (scroll container added as groundwork)
- No dedicated visual cluster board yet (future extension over current preview primitives)
