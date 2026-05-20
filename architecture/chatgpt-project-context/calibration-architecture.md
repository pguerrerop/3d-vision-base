# Calibration Architecture

## Scope

Current calibration supports two independent tracks:

- `plane_3d`: point-cloud belt plane calibration (existing)
- `camera_2d`: RGB planar belt calibration for pixel-to-mm measurement (new)

This phase is strictly planar belt geometry. No height compensation or full 3D reprojection correction is applied.

## Camera 2D model

`camera_2d` calibration persists under `config/calibrations/*.json` with:

- target config (`charuco` primary, `checkerboard` secondary)
- camera intrinsics (`camera_matrix`, `dist_coeffs`, reprojection error)
- belt plane homography (`pixel -> mm` on belt plane)
- diagnostics scale (`mm_per_px_x`, `mm_per_px_y`)

## Execution integration

2D ProcessService loads the active runtime default calibration when available. If it is `camera_2d`, ellipse metrics include:

- `equivalent_diameter_mm`
- `major_axis_mm`
- `minor_axis_mm`

If no active 2D calibration exists, mm fields remain `null` and px metrics remain authoritative.

## UI architecture

Calibration page tabs:

- `3D Plane`
- `2D Camera`
- `Laser Line` (placeholder)
- `Fusion` (placeholder)

The 2D tab includes capture, corner detection, calibration solve, diagnostics, save/load, and active/default assignment.

## Forward path

The model intentionally leaves room for:

- explicit camera extrinsics
- perspective/height compensation
- RGB + 3D alignment
- multi-camera fusion
- true sphere estimation with height-aware correction

## Source discovery and freshness semantics

2D camera calibration now resolves source selection from `GET /api/sources` instead of raw source-id text entry.

Source payload fields:

- `id` (stable internal identifier)
- `label` (friendly display label)
- `type`
- `modality`
- `status` (`live | stale | unavailable`)
- `last_frame_at`
- `last_frame_age_seconds`
- `resolution`
- `fps`
- runtime active/session hints when available

Freshness threshold for calibration preview is 5 seconds. Stale previews are explicitly labeled and visually dimmed to avoid accidental calibration on old frames.

## Capture workflow safety

2D calibration capture flow:

1. Select source from discovered list.
2. If preview is stale, user is prompted to refresh.
3. Refresh frame captures a new preview frame from the source when supported.
4. Capture calibration frame persists only after freshness checks.
5. Detect corners and calibrate remain disabled until at least one fresh capture exists and target parameters are valid.

## Capture-first 2D workflow (Option A)

2D calibration now follows a capture-first architecture:

`Select source -> Capture calibration frame -> Detect corners -> Calibrate -> Save`

Key semantics:

- Calibration inputs are persisted captures, not transient live preview frames.
- Preview freshness is informational-only and never blocks capture.
- Capture action acquires a fresh frame directly from source, stores it, and updates preview.

Captured frames are persisted under:

- `data/calibration/captures/`

Each capture stores source id, timestamp, image path, resolution, freshness flag, and corner-detection metadata.
