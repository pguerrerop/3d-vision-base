#!/usr/bin/env python3
"""Capture RGB image or video takes from a local USB camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.acquisition.usb_camera import capture_image, record_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture USB camera RGB takes.")
    parser.add_argument("--camera-index", default=0, type=int, help="OpenCV camera index")
    parser.add_argument("--mode", choices=("image", "video"), required=True, help="Capture mode")
    parser.add_argument("--duration", default=10.0, type=float, help="Video duration in seconds")
    parser.add_argument("--data-dir", default=Path("data"), type=Path, help="Data root")
    parser.add_argument("--session", default=None, help="Optional session id")
    parser.add_argument("--preview-window", action="store_true", help="Show optional OpenCV/native preview window")
    parser.add_argument("--preview", dest="preview_window", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preview-interval-ms", default=250, type=int, help="Browser preview JPEG export interval")
    parser.add_argument("--width", default=None, type=int, help="Requested camera width")
    parser.add_argument("--height", default=None, type=int, help="Requested camera height")
    parser.add_argument("--fps", default=None, type=float, help="Requested camera/video FPS")
    parser.add_argument("--output-name", default=None, help="Optional take id/output folder name")
    args = parser.parse_args()

    try:
        if args.mode == "image":
            take_id, folder = capture_image(
                data_dir=args.data_dir,
                camera_index=args.camera_index,
                session_id=args.session,
                width=args.width,
                height=args.height,
                fps=args.fps,
                preview_window=args.preview_window,
                preview_interval_ms=args.preview_interval_ms,
                output_name=args.output_name,
            )
        else:
            take_id, folder = record_video(
                data_dir=args.data_dir,
                camera_index=args.camera_index,
                duration=args.duration,
                session_id=args.session,
                width=args.width,
                height=args.height,
                fps=args.fps,
                preview_window=args.preview_window,
                preview_interval_ms=args.preview_interval_ms,
                output_name=args.output_name,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"take_id: {take_id}")
    print(f"folder: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
