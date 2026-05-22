from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.apps.ball_inspection_25d.pipeline import run_ball_inspection_25d_flow


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 25D cube calibration diagnostics.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--take-id", required=True)
    parser.add_argument("--known-width-mm", type=float, required=True)
    parser.add_argument("--known-depth-mm", type=float, required=True)
    parser.add_argument("--known-height-mm", type=float, required=True)
    parser.add_argument("--tolerance-percent", type=float, default=5.0)
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    take_dir = data_dir / "incoming" / args.take_id
    metadata_path = take_dir / "metadata.json"
    metadata = _read_json(metadata_path) or {}
    metadata["known_object_25d"] = {
        "enabled": True,
        "object_label": "cubo",
        "known_width_mm": args.known_width_mm,
        "known_depth_mm": args.known_depth_mm,
        "known_height_mm": args.known_height_mm,
        "tolerance_percent": args.tolerance_percent,
        "target_selection": "largest_component",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    result = run_ball_inspection_25d_flow(data_dir, take_id=args.take_id)
    out = result.output_dir
    plane_debug = _read_json(out / "plane_fit_debug.json") or {}
    norm_debug = _read_json(out / "normalized_height_debug.json") or {}
    scale = _read_json(out / "known_object_scale_validation.json") or {}

    print("run_path:", out)
    print("plane_fit_status:", plane_debug.get("status"))
    print("plane_inlier_ratio:", plane_debug.get("inlier_ratio"))
    print(
        "plane_residuals_mm:",
        {
            "mean": plane_debug.get("residual_mean_mm"),
            "median": plane_debug.get("residual_median_mm"),
            "p95_abs": plane_debug.get("residual_p95_mm"),
            "max_abs": plane_debug.get("residual_max_abs_mm"),
        },
    )
    print(
        "background_normalized_stats_mm:",
        {
            "mean": norm_debug.get("background_height_mean_after_normalization_mm"),
            "p95_abs": norm_debug.get("background_height_p95_abs_after_normalization_mm"),
        },
    )
    print(
        "cube_measured_mm:",
        {
            "width": scale.get("measured_width_mm"),
            "depth": scale.get("measured_depth_mm"),
            "height": scale.get("measured_height_mm"),
        },
    )
    print(
        "scale_error_percent:",
        {
            "x": scale.get("scale_error_x_percent"),
            "y": scale.get("scale_error_y_percent"),
            "z": scale.get("scale_error_z_percent"),
        },
    )
    print(
        "recommended_scale_corrections:",
        {
            "x": scale.get("recommended_scale_correction_x"),
            "y": scale.get("recommended_scale_correction_y"),
            "z": scale.get("recommended_scale_correction_z"),
        },
    )
    print(
        "artifacts:",
        {
            "plane_fit_debug": str(out / "plane_fit_debug.json"),
            "normalized_height_histogram": str(out / "normalized_height_histogram.json"),
            "known_object_scale_validation": str(out / "known_object_scale_validation.json"),
        },
    )


if __name__ == "__main__":
    main()
