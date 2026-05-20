from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.state.runtime_status import update_runtime_status

PREVIEW_STALE_SECONDS = 3.0


@dataclass(frozen=True)
class PreviewMetadata:
    source: str
    camera_index: int | None
    timestamp: str
    resolution: list[int] | None
    stale: bool
    fps_estimate: float | None = None
    path: str | None = None
    updated_at_monotonic: float | None = None
    error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "camera_index": self.camera_index,
            "timestamp": self.timestamp,
            "resolution": self.resolution,
            "stale": self.stale,
            "fps_estimate": self.fps_estimate,
            "path": self.path,
            "error": self.error,
        }


class PreviewWriter:
    def __init__(
        self,
        data_dir: Path,
        *,
        source: str,
        camera_index: int | None = None,
        interval_ms: int = 250,
        stale_seconds: float = PREVIEW_STALE_SECONDS,
        jpeg_quality: int = 80,
        cv2_module: Any,
    ) -> None:
        self.data_dir = data_dir
        self.source = source
        self.camera_index = camera_index
        self.interval_seconds = max(interval_ms, 1) / 1000.0
        self.stale_seconds = stale_seconds
        self.jpeg_quality = int(max(1, min(jpeg_quality, 100)))
        self.cv2 = cv2_module
        self.preview_dir = data_dir / "runtime" / "previews"
        self.preview_path = self.preview_dir / _preview_filename(source, camera_index)
        self.metadata_path = self.preview_path.with_suffix(".json")
        self._last_export_monotonic = 0.0
        self._last_success_monotonic: float | None = None
        self._export_count = 0
        self._started = time.monotonic()

    def maybe_export(self, frame: Any) -> bool:
        now = time.monotonic()
        if now - self._last_export_monotonic < self.interval_seconds:
            return False
        self._last_export_monotonic = now
        self.export(frame)
        return True

    def export(self, frame: Any) -> PreviewMetadata:
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).isoformat()
        height, width = frame.shape[:2]
        params = [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality] if hasattr(self.cv2, "IMWRITE_JPEG_QUALITY") else []
        ok = self.cv2.imwrite(str(self.preview_path), frame, params)
        if not ok:
            metadata = self._metadata(
                timestamp=timestamp,
                resolution=[int(width), int(height)],
                stale=True,
                error="preview export failed",
            )
            _write_json(self.metadata_path, metadata.model_dump())
            self._update_runtime(metadata, warning="preview export failed")
            raise RuntimeError("Failed to export latest preview frame.")
        self._export_count += 1
        self._last_success_monotonic = time.monotonic()
        elapsed = max(self._last_success_monotonic - self._started, 0.001)
        metadata = self._metadata(
            timestamp=timestamp,
            resolution=[int(width), int(height)],
            stale=False,
            fps_estimate=self._export_count / elapsed,
        )
        _write_json(self.metadata_path, metadata.model_dump())
        self._update_runtime(metadata)
        return metadata

    def mark_disconnected(self, message: str) -> PreviewMetadata:
        metadata = self._metadata(
            timestamp=datetime.now(UTC).isoformat(),
            resolution=None,
            stale=True,
            error=message,
        )
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.metadata_path, metadata.model_dump())
        self._update_runtime(metadata, warning=message)
        return metadata

    def _metadata(
        self,
        *,
        timestamp: str,
        resolution: list[int] | None,
        stale: bool,
        fps_estimate: float | None = None,
        error: str | None = None,
    ) -> PreviewMetadata:
        return PreviewMetadata(
            source=self.source,
            camera_index=self.camera_index,
            timestamp=timestamp,
            resolution=resolution,
            stale=stale,
            fps_estimate=round(float(fps_estimate), 3) if fps_estimate is not None else None,
            path=str(self.preview_path),
            error=error,
        )

    def _update_runtime(self, metadata: PreviewMetadata, warning: str | None = None) -> None:
        warnings = [warning] if warning else []
        update_runtime_status(
            self.data_dir / "state",
            preview_available=self.preview_path.is_file(),
            preview_timestamp=metadata.timestamp,
            preview_fps_estimate=metadata.fps_estimate,
            preview_stale=metadata.stale,
            preview_source=metadata.source,
            preview_path=str(self.preview_path),
            warnings=warnings,
        )


def preview_image_path(data_dir: Path, source_id: str | None = None) -> Path:
    if source_id:
        return data_dir / "runtime" / "previews" / f"{source_id}.jpg"
    return data_dir / "runtime" / "previews" / "usb_camera_0.jpg"


def preview_metadata_path(data_dir: Path, source_id: str | None = None) -> Path:
    return preview_image_path(data_dir, source_id).with_suffix(".json")


def read_preview_metadata(data_dir: Path, source_id: str | None = None, *, stale_seconds: float = PREVIEW_STALE_SECONDS) -> dict[str, Any]:
    path = preview_metadata_path(data_dir, source_id)
    if not path.is_file():
        return {
            "source": source_id or "usb_camera",
            "camera_index": 0 if source_id is None else None,
            "timestamp": None,
            "resolution": None,
            "stale": True,
            "fps_estimate": None,
            "path": str(preview_image_path(data_dir, source_id)),
            "error": "no preview available",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        payload = {"error": "preview metadata unreadable"}
    if not isinstance(payload, dict):
        payload = {"error": "preview metadata invalid"}
    payload.setdefault("source", source_id or "usb_camera")
    payload.setdefault("camera_index", 0 if source_id is None else None)
    payload.setdefault("path", str(preview_image_path(data_dir, source_id)))
    payload["stale"] = bool(payload.get("stale")) or _is_timestamp_stale(payload.get("timestamp"), stale_seconds=stale_seconds)
    return payload


def _preview_filename(source: str, camera_index: int | None) -> str:
    suffix = "unknown" if camera_index is None else str(camera_index)
    return f"{source}_{suffix}.jpg"


def _is_timestamp_stale(timestamp: Any, *, stale_seconds: float) -> bool:
    if not timestamp:
        return True
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds() > stale_seconds


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
