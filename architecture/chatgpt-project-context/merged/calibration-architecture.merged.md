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

## Added source discovery + freshness gating

2D calibration no longer relies on manual source-id text input. It uses discovered sources with friendly labels and explicit freshness metadata, and gates detect/calibrate actions on fresh captured frames and valid target configuration.

Capture-first refactor: 2D calibration no longer depends on preview freshness. Calibration operates on persisted capture IDs stored in `data/calibration/captures/`, and capture acquisition fetches a fresh source frame directly.
