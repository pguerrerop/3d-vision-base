#!/usr/bin/env python3
"""List local USB cameras using OpenCV VideoCapture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.acquisition.usb_camera import discover_cameras, format_camera_info  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="List local USB cameras.")
    parser.add_argument("--max-index", default=8, type=int, help="Highest camera index to probe")
    args = parser.parse_args()

    try:
        cameras = discover_cameras(max_index=args.max_index)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for camera in cameras:
        print(format_camera_info(camera))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
