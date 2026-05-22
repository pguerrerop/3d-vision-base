from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import cv2

from vision_3d_acquisition.acquisition.preview import preview_image_path
from vision_3d_acquisition.acquisition.usb_camera import capture_single_frame


def mjpeg_stream_frames(*, data_dir: Path, source_id: str, fps: float = 8.0) -> Iterator[bytes]:
    interval = max(0.05, 1.0 / max(1.0, float(fps)))
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = _acquire_frame(data_dir=data_dir, source_id=source_id)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield boundary + encoded.tobytes() + b"\r\n"
        time.sleep(interval)


def _acquire_frame(*, data_dir: Path, source_id: str) -> Any:
    if source_id.startswith("usb_camera_"):
        camera_index = int(source_id.rsplit("_", 1)[1])
        capture = capture_single_frame(data_dir=data_dir, camera_index=camera_index, session_id=None)
        return capture.frame
    preview_path = preview_image_path(data_dir, source_id)
    if not preview_path.is_file():
        raise RuntimeError(f"Preview not available for source: {source_id}")
    image = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode preview image for source: {source_id}")
    return image
