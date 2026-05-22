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
2. Capture calibration frame (primary action) acquires a fresh frame directly from source.
3. Preview-cache fallback is secondary only and must satisfy freshness threshold.
4. Detect corners runs on selected capture by default (`Detect all captures` is explicit secondary action).
5. Calibrate remains gated by successful corner detections, not preview freshness.

## Capture-first 2D workflow (Option A)

2D calibration now follows a capture-first architecture:

`Select source -> Capture calibration frame -> Detect corners -> Calibrate -> Save`

Key semantics:

- Calibration inputs are persisted captures, not transient live preview frames.
- Preview freshness is informational-only and never blocks capture.
- Capture action acquires a fresh frame directly from source, stores it, and updates preview.

Captured frames are persisted under:

- `data/calibration/captures/`

Each capture stores source id, timestamp, image path, resolution, freshness flag, and corner-detection metadata, including dictionary used and detection diagnostics.

Supported ChArUco dictionaries:

- `DICT_4X4_50`
- `DICT_4X4_100`
- `DICT_5X5_50`
- `DICT_5X5_100`
- `DICT_6X6_250`

## Camera runtime controls (calibration tuning)

2D Camera tab now exposes runtime camera controls for USB sources:

- auto exposure + exposure
- auto focus + focus
- gain, brightness
- contrast, sharpness, saturation, white balance (when supported)

API:

- `GET /api/sources/{source_id}/controls`
- `POST /api/sources/{source_id}/controls`

Unsupported controls degrade gracefully and are shown as unavailable.

Calibration persistence includes `camera_runtime_settings` so imaging conditions used during calibration can be restored later.

## Modal architecture refactor

2D Camera calibration UX is now split into three responsibilities:

- Main Calibration Manager page: calibration assets, active/default assignment, summary metrics.
- Camera Controls modal: realtime MJPEG stream, runtime camera tuning, snapshot capture.
- ChArUco Calibration modal: board config, detection/debug overlays, calibration solve/save.

Realtime viewing is moved out of the manager page to reduce stale/live preview confusion.

MJPEG endpoint:

- `GET /api/runtime/stream/mjpeg?source_id=<id>&fps=<n>`

This stream is intended for low-latency operator tuning (5–15 FPS initial target), while calibration continues to be capture-first over persisted snapshots.
