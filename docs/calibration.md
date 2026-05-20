# Calibration Workflow

Calibration is a web-based operator workflow for giving semantic meaning to geometry that the system detects automatically. The operator does not draw polygons, pick arbitrary 3D points, or manually define planes. Plane polygons shown in the UI are inferred from detected plane candidate geometry.

## Workflow

1. Open `/calibration`.
2. Select a processed reference take.
3. Run plane detection.
4. Review the combined preview and one preview per candidate plane.
5. Assign each detected plane one semantic label:
   - `belt`: the conveyor or reference support plane.
   - `outer_plane_ignore`: fixed surrounding geometry that should be removed before clustering.
   - `unused`: detected geometry that should not affect processing.
6. Save the calibration.

The saved file is written to `config/calibrations/` as JSON and can be loaded by later real processing runs.

## Plane Candidates

Plane candidates are extracted with iterative RANSAC. The dominant plane is detected first, its inlier points are removed, and the process repeats until the requested limit is reached or too few plane points remain.

Each candidate stores the plane equation, point count, centroid, bounding box, extent, average Z, normal vector, and `roi_polygon_xy_mm`. The polygon is computed by projecting the plane inlier points to XY and taking a convex hull. If a candidate is degenerate, the system falls back to a rectangle from the candidate bounding box.

## Units

Geometry values use millimeters. Field names in JSON keep the `_mm` suffix where relevant:

- `bbox_min_mm`, `bbox_max_mm`, `extent_mm`, `avg_z_mm`
- `roi_polygon_xy_mm` as `[[x_mm, y_mm], ...]`
- object centers, dimensions, diameter estimates, and height filters in millimeters

Plane normals are unitless direction vectors. Point counts are counts, not physical units.

## Belt Plane

Exactly one plane must be labeled `belt`. The belt plane becomes the reference surface for foreground filtering. Object heights are evaluated relative to this plane, and belt area checks use `roi_polygon_xy_mm` when present. The bounding box remains a fallback for older or incomplete calibrations.

## Ignored Outer Planes

Any number of planes may be labeled `outer_plane_ignore`. During calibrated processing, points close to these planes are removed before foreground clustering. This is meant for fixed side rails, walls, supports, or other scene geometry that should not become object clusters.

## Object Filtering

Each calibration contains an `object_filter` block:

```json
{
  "min_height_above_belt_mm": 3,
  "max_height_above_belt_mm": 130,
  "require_center_inside_belt": true,
  "min_fraction_points_inside_belt": 0.6
}
```

The processing pipeline uses these values to keep foreground points near and above the belt, then keeps only clusters that satisfy the belt polygon rules. It checks both the cluster center inside the belt polygon and the fraction of cluster points inside the belt polygon. If no polygon is available, the belt bounding box is used.

## Processing

Run calibrated processing with:

```bash
python scripts/process_latest_real.py --data-dir data --calibration config/calibrations/belt_setup_2026_05_16.json
```

Set the active default calibration in the Calibration UI (**Set as active/default**), via `POST /api/runtime-config/default-calibration`, or with:

```bash
python scripts/process_latest_real.py --set-default-calibration config/calibrations/belt_setup_2026_05_16.json
```

That writes `default_calibration_file` to `config/runtime.json`. Copy `config/runtime.json.example` to `config/runtime.json` when setting up a new environment.

When calibrated processing is used, processed result JSON includes `calibration_id`, and the Debug page shows a calibration badge.

Calibrated processing differs from generic mode. Generic mode detects one dominant plane at processing time. Calibrated mode uses the saved `belt` plane as the reference, removes saved `outer_plane_ignore` planes, filters points by height above the belt, and filters objects by the saved belt polygon.

## Lifecycle

A calibration should be regenerated when the belt, sensor pose, fixed surrounding planes, or processing region changes. Reference takes should be empty or representative of the static environment so the detected planes describe the environment rather than objects.
