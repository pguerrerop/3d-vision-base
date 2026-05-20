#!/usr/bin/env python3
"""Run reusable acquisition/debug flow without ball-specific assumptions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.acquisition_studio import run_acquisition_debug_flow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run acquisition studio debug pipeline.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data root")
    parser.add_argument("--take-id", default=None, help="Specific capture/take id")
    parser.add_argument("--calibration", default=None, type=Path, help="Calibration JSON")
    parser.add_argument(
        "--prefer-fast-cloud",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prefer point_cloud.npz runtime input when present",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip point cloud preview rendering")
    parser.add_argument("--profile", action="store_true", help="Print profiling stages")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    try:
        result = run_acquisition_debug_flow(
            data_dir,
            take_id=args.take_id,
            calibration=args.calibration,
            prefer_fast_cloud=args.prefer_fast_cloud,
            render_preview=not args.no_preview,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"take_id: {result.take_id}")
    print(f"output_dir: {result.output_dir}")
    print(f"cluster_count: {result.cluster_count}")
    if result.plane_model is not None:
        print(f"plane_model: {result.plane_model}")
    if args.profile:
        print("profiling_stages:")
        for stage in result.profiling.get("stages", []):
            print(f"  - {stage['name']}: {stage['duration_ms']:.2f} ms ({stage['category']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
