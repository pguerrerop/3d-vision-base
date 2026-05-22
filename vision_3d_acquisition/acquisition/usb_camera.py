from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.acquisition.preview import PreviewWriter
from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata, FileReferences, FrameSet
from vision_3d_acquisition.state.runtime_status import update_runtime_status
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher
from vision_3d_acquisition.utils.ids import generate_take_id
from vision_3d_acquisition.utils.time import utc_now_iso


@dataclass(frozen=True)
class CameraInfo:
    index: int
    working: bool
    width: int | None = None
    height: int | None = None
    backend: str | None = None
    name: str | None = None
    error: str | None = None

    def label(self) -> str:
        return self.name or f"Camera {self.index}"

    def resolution(self) -> list[int] | None:
        if self.width and self.height:
            return [self.width, self.height]
        return None

    def model_dump(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "working": self.working,
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
            "name": self.name,
            "error": self.error,
        }


@dataclass(frozen=True)
class UsbFrameCapture:
    frame: Any
    camera_index: int
    backend: str | None
    resolution: list[int] | None
    fps: float | None
    timestamp: str


CONTROL_PROP_MAP: dict[str, str] = {
    "exposure": "CAP_PROP_EXPOSURE",
    "focus": "CAP_PROP_FOCUS",
    "gain": "CAP_PROP_GAIN",
    "brightness": "CAP_PROP_BRIGHTNESS",
    "contrast": "CAP_PROP_CONTRAST",
    "sharpness": "CAP_PROP_SHARPNESS",
    "saturation": "CAP_PROP_SATURATION",
    "white_balance": "CAP_PROP_WHITE_BALANCE_BLUE_U",
}

CONTROL_AUTO_PROP_MAP: dict[str, str] = {
    "exposure": "CAP_PROP_AUTO_EXPOSURE",
    "focus": "CAP_PROP_AUTOFOCUS",
}

CONTROL_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "exposure": (-13.0, 0.0),
    "focus": (0.0, 255.0),
    "gain": (0.0, 255.0),
    "brightness": (0.0, 255.0),
    "contrast": (0.0, 255.0),
    "sharpness": (0.0, 255.0),
    "saturation": (0.0, 255.0),
    "white_balance": (2000.0, 9000.0),
}

CONTROL_DEFAULT_STEPS: dict[str, float] = {
    "exposure": 0.1,
    "focus": 1.0,
    "gain": 1.0,
    "brightness": 1.0,
    "contrast": 1.0,
    "sharpness": 1.0,
    "saturation": 1.0,
    "white_balance": 100.0,
}

CONTROL_UNITS: dict[str, str | None] = {
    "exposure": "ev",
    "focus": "level",
    "gain": "level",
    "brightness": "level",
    "contrast": "level",
    "sharpness": "level",
    "saturation": "level",
    "white_balance": "kelvin",
}


def discover_cameras(max_index: int = 8, cv2_module: Any | None = None) -> list[CameraInfo]:
    cv2 = _cv2(cv2_module)
    cameras: list[CameraInfo] = []
    for index in range(max_index + 1):
        cap = None
        try:
            cap = cv2.VideoCapture(index)
            if not cap or not cap.isOpened():
                cameras.append(CameraInfo(index=index, working=False, error="unavailable"))
                continue
            ok, frame = cap.read()
            backend = _backend_name(cap)
            width, height = _frame_resolution(frame) if ok else _capture_resolution(cap)
            cameras.append(
                CameraInfo(
                    index=index,
                    working=bool(ok),
                    width=width,
                    height=height,
                    backend=backend,
                    error=None if ok else "open succeeded but read failed",
                )
            )
        except Exception as exc:
            cameras.append(CameraInfo(index=index, working=False, error=str(exc)))
        finally:
            if cap is not None:
                cap.release()
    return cameras


def format_camera_info(camera: CameraInfo) -> str:
    if not camera.working:
        return f"[{camera.index}] unavailable"
    resolution = f"{camera.width}x{camera.height}" if camera.width and camera.height else "unknown resolution"
    suffix = f" ({camera.backend})" if camera.backend else ""
    return f"[{camera.index}] {camera.label()} - OK - {resolution}{suffix}"


def capture_image(
    *,
    data_dir: Path,
    camera_index: int,
    session_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    preview_window: bool = False,
    preview_interval_ms: int = 250,
    output_name: str | None = None,
    acquisition_group_id: str | None = None,
    cv2_module: Any | None = None,
) -> tuple[str, Path]:
    cv2 = _cv2(cv2_module)
    if not preview_window:
        capture = capture_single_frame(
            data_dir=data_dir,
            camera_index=camera_index,
            session_id=session_id,
            width=width,
            height=height,
            fps=fps,
            preview_interval_ms=preview_interval_ms,
            cv2_module=cv2,
        )
        take_id = output_name or generate_take_id()
        created_at = utc_now_iso()
        with tempfile.TemporaryDirectory(prefix="usb_camera_capture_") as temp:
            rgb_path = Path(temp) / "rgb.png"
            if not cv2.imwrite(str(rgb_path), capture.frame):
                raise RuntimeError("Failed to write RGB image.")
            metadata = AcquisitionMetadata(
                take_id=take_id,
                source={
                    "type": "usb_camera",
                    "camera_index": camera_index,
                    "backend": capture.backend,
                    "resolution": capture.resolution,
                    "fps": round(float(capture.fps), 3) if capture.fps is not None else None,
                },
                mode="live",
                created_at=created_at,
                frame_count=1,
                modalities=["rgb"],  # type: ignore[arg-type]
                session_id=session_id,
                acquisition_group_id=acquisition_group_id,
                frameset=FrameSet(
                    frameset_id=f"{take_id}_fs0",
                    timestamp=created_at,
                    frame_count=1,
                    synchronized=False,
                    timestamp_source="usb_camera",
                ),
                files=FileReferences(rgb="rgb.png"),
            )
            folder = AcquisitionPublisher(data_dir).publish_take(take_id, metadata, {"rgb": rgb_path})
        return take_id, folder

    with _open_camera(cv2, camera_index, width=width, height=height, fps=fps, data_dir=data_dir, session_id=session_id) as cap:
        preview_writer = PreviewWriter(
            data_dir,
            source="usb_camera",
            camera_index=camera_index,
            interval_ms=preview_interval_ms,
            cv2_module=cv2,
        )
        frame, measured_fps = _preview_until_capture(cv2, cap, camera_index, session_id, preview_writer, enabled=True)
        preview_writer.export(frame)
        take_id = output_name or generate_take_id()
        created_at = utc_now_iso()
        with tempfile.TemporaryDirectory(prefix="usb_camera_capture_") as temp:
            rgb_path = Path(temp) / "rgb.png"
            if not cv2.imwrite(str(rgb_path), frame):
                raise RuntimeError("Failed to write RGB image.")
            metadata = _metadata(
                take_id=take_id,
                camera_index=camera_index,
                cap=cap,
                created_at=created_at,
                session_id=session_id,
                acquisition_group_id=acquisition_group_id,
                frame_count=1,
                files=FileReferences(rgb="rgb.png"),
                modalities=["rgb"],
                fps=measured_fps or fps,
            )
            folder = AcquisitionPublisher(data_dir).publish_take(take_id, metadata, {"rgb": rgb_path})
        return take_id, folder


def capture_single_frame(
    *,
    data_dir: Path,
    camera_index: int,
    session_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    preview_interval_ms: int = 250,
    cv2_module: Any | None = None,
) -> UsbFrameCapture:
    cv2 = _cv2(cv2_module)
    with _open_camera(cv2, camera_index, width=width, height=height, fps=fps, data_dir=data_dir, session_id=session_id) as cap:
        preview_writer = PreviewWriter(
            data_dir,
            source="usb_camera",
            camera_index=camera_index,
            interval_ms=preview_interval_ms,
            cv2_module=cv2,
        )
        frame, measured_fps = _read_frame(cap)
        preview_writer.export(frame)
        width_px, height_px = _frame_resolution(frame)
        return UsbFrameCapture(
            frame=frame,
            camera_index=camera_index,
            backend=_backend_name(cap),
            resolution=[width_px, height_px] if width_px and height_px else None,
            fps=measured_fps,
            timestamp=datetime.now(UTC).isoformat(),
        )


def record_video(
    *,
    data_dir: Path,
    camera_index: int,
    duration: float,
    session_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    preview_window: bool = False,
    preview_interval_ms: int = 250,
    output_name: str | None = None,
    acquisition_group_id: str | None = None,
    cv2_module: Any | None = None,
) -> tuple[str, Path]:
    cv2 = _cv2(cv2_module)
    with _open_camera(cv2, camera_index, width=width, height=height, fps=fps, data_dir=data_dir, session_id=session_id) as cap:
        take_id = output_name or generate_take_id()
        created_at = utc_now_iso()
        target_fps = fps or _capture_fps(cap) or 30.0
        preview_writer = PreviewWriter(
            data_dir,
            source="usb_camera",
            camera_index=camera_index,
            interval_ms=preview_interval_ms,
            cv2_module=cv2,
        )
        with tempfile.TemporaryDirectory(prefix="usb_camera_video_") as temp:
            temp_dir = Path(temp)
            first_frame, _ = _read_frame(cap)
            preview_writer.export(first_frame)
            preview_path = temp_dir / "preview.png"
            if not cv2.imwrite(str(preview_path), first_frame):
                raise RuntimeError("Failed to write video preview image.")
            video_path, writer = _video_writer(cv2, temp_dir, first_frame, target_fps)
            frame_count = 0
            started = time.monotonic()
            try:
                frame_count += _write_video_frame(cv2, writer, first_frame, preview_window, preview_writer, cap, camera_index, session_id, frame_count)
                while time.monotonic() - started < duration:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_count += _write_video_frame(cv2, writer, frame, preview_window, preview_writer, cap, camera_index, session_id, frame_count)
                    if preview_window and cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            finally:
                writer.release()
                if preview_window:
                    cv2.destroyAllWindows()
            elapsed = max(time.monotonic() - started, 0.001)
            metadata = _metadata(
                take_id=take_id,
                camera_index=camera_index,
                cap=cap,
                created_at=created_at,
                session_id=session_id,
                acquisition_group_id=acquisition_group_id,
                frame_count=max(frame_count, 1),
                files=FileReferences(rgb="preview.png", rgb_video=video_path.name),
                modalities=["rgb", "rgb_video"],
                fps=frame_count / elapsed,
                duration=elapsed,
            )
            folder = AcquisitionPublisher(data_dir).publish_take(
                take_id,
                metadata,
                {"rgb": preview_path, "rgb_video": video_path},
            )
        return take_id, folder


class _CameraHandle:
    def __init__(self, cv2: Any, cap: Any, data_dir: Path, session_id: str | None, camera_index: int) -> None:
        self.cv2 = cv2
        self.cap = cap
        self.data_dir = data_dir
        self.session_id = session_id
        self.camera_index = camera_index

    def __enter__(self) -> Any:
        return self.cap

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.cap.release()
        if exc is not None:
            warnings = [f"USB camera {self.camera_index} failed: {exc}"]
            update_runtime_status(
                self.data_dir / "state",
                status="idle",
                acquisition_connected=False,
                current_session=self.session_id,
                acquisition_source="usb_camera",
                acquisition_source_details={"type": "usb_camera", "camera_index": self.camera_index},
                warnings=warnings,
                message=warnings[0],
            )


def _open_camera(
    cv2: Any,
    camera_index: int,
    *,
    width: int | None,
    height: int | None,
    fps: float | None,
    data_dir: Path,
    session_id: str | None,
) -> _CameraHandle:
    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        if cap is not None:
            cap.release()
        update_runtime_status(
            data_dir / "state",
            status="idle",
            acquisition_connected=False,
            current_session=session_id,
            acquisition_source="usb_camera",
            acquisition_source_details={"type": "usb_camera", "camera_index": camera_index},
            warnings=[f"USB camera {camera_index} is unavailable."],
            message=f"USB camera {camera_index} is unavailable.",
        )
        raise RuntimeError(f"Camera index {camera_index} is unavailable.")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    update_runtime_status(
        data_dir / "state",
        status="acquiring",
        acquisition_connected=True,
        current_session=session_id,
        acquisition_source="usb_camera",
        acquisition_source_details={"type": "usb_camera", "camera_index": camera_index, "backend": _backend_name(cap)},
        message="USB camera connected.",
    )
    return _CameraHandle(cv2, cap, data_dir, session_id, camera_index)


def _read_frame(cap: Any) -> tuple[Any, float | None]:
    started = time.monotonic()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but no frame could be read.")
    elapsed = time.monotonic() - started
    return frame, (1.0 / elapsed if elapsed > 0 else None)


def _preview_until_capture(
    cv2: Any,
    cap: Any,
    camera_index: int,
    session_id: str | None,
    preview_writer: PreviewWriter,
    *,
    enabled: bool,
) -> tuple[Any, float | None]:
    last = time.monotonic()
    measured_fps: float | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera opened but preview frame could not be read.")
        now = time.monotonic()
        measured_fps = 1.0 / max(now - last, 0.001)
        last = now
        preview_writer.maybe_export(frame)
        _draw_overlay(cv2, frame, camera_index, session_id, measured_fps, recording=False)
        if enabled:
            cv2.imshow("USB camera preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cv2.destroyAllWindows()
                return frame, measured_fps
            if key == ord("q"):
                cv2.destroyAllWindows()
                raise RuntimeError("Preview quit before capture.")
        else:
            return frame, measured_fps


def _write_video_frame(
    cv2: Any,
    writer: Any,
    frame: Any,
    preview_window: bool,
    preview_writer: PreviewWriter,
    cap: Any,
    camera_index: int,
    session_id: str | None,
    frame_count: int,
) -> int:
    writer.write(frame)
    preview_writer.maybe_export(frame)
    if preview_window:
        _draw_overlay(cv2, frame, camera_index, session_id, _capture_fps(cap), recording=True)
        cv2.imshow("USB camera recording", frame)
    return 1


def _draw_overlay(cv2: Any, frame: Any, camera_index: int, session_id: str | None, fps: float | None, *, recording: bool) -> None:
    height, width = frame.shape[:2]
    text = f"USB {camera_index} {width}x{height}"
    if fps:
        text += f" {fps:.1f} fps"
    if session_id:
        text += f" session={session_id}"
    if recording:
        text += " REC"
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 240, 30), 2, cv2.LINE_AA)


def _video_writer(cv2: Any, temp_dir: Path, frame: Any, fps: float) -> tuple[Path, Any]:
    height, width = frame.shape[:2]
    for filename, codec in (("rgb_video.mp4", "mp4v"), ("rgb_video.avi", "MJPG")):
        path = temp_dir / filename
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return path, writer
        writer.release()
    raise RuntimeError("No usable video codec found for RGB video capture.")


def _metadata(
    *,
    take_id: str,
    camera_index: int,
    cap: Any,
    created_at: str,
    session_id: str | None,
    acquisition_group_id: str | None,
    frame_count: int,
    files: FileReferences,
    modalities: list[str],
    fps: float | None = None,
    duration: float | None = None,
) -> AcquisitionMetadata:
    width, height = _capture_resolution(cap)
    source = {
        "type": "usb_camera",
        "camera_index": camera_index,
        "backend": _backend_name(cap),
        "resolution": [width, height] if width and height else None,
    }
    if fps is not None:
        source["fps"] = round(float(fps), 3)
    if duration is not None:
        source["duration_seconds"] = round(float(duration), 3)
    return AcquisitionMetadata(
        take_id=take_id,
        source=source,
        mode="live",
        created_at=created_at,
        frame_count=frame_count,
        modalities=modalities,  # type: ignore[arg-type]
        session_id=session_id,
        acquisition_group_id=acquisition_group_id,
        frameset=FrameSet(
            frameset_id=f"{take_id}_fs0",
            timestamp=created_at,
            frame_count=frame_count,
            synchronized=False,
            timestamp_source="usb_camera",
        ),
        files=files,
    )


def _cv2(cv2_module: Any | None = None) -> Any:
    if cv2_module is not None:
        return cv2_module
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for USB camera acquisition.") from exc
    return cv2


def _backend_name(cap: Any) -> str | None:
    getter = getattr(cap, "getBackendName", None)
    if not callable(getter):
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _capture_resolution(cap: Any) -> tuple[int | None, int | None]:
    try:
        width = int(cap.get(3))
        height = int(cap.get(4))
    except Exception:
        return None, None
    return (width or None), (height or None)


def _capture_fps(cap: Any) -> float | None:
    try:
        fps = float(cap.get(5))
    except Exception:
        return None
    return fps if fps > 0 else None


def _frame_resolution(frame: Any) -> tuple[int | None, int | None]:
    if frame is None or not hasattr(frame, "shape"):
        return None, None
    height, width = frame.shape[:2]
    return int(width), int(height)


def camera_controls_snapshot(camera_index: int, cv2_module: Any | None = None) -> dict[str, Any]:
    cv2 = _cv2(cv2_module)
    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise RuntimeError(f"Camera index {camera_index} is unavailable.")
    try:
        backend = _backend_name(cap)
        controls: dict[str, Any] = {}
        raw_get: dict[str, Any] = {}
        unsupported: list[str] = []
        for name, prop_name in CONTROL_PROP_MAP.items():
            prop_id = getattr(cv2, prop_name, None)
            auto_prop_name = CONTROL_AUTO_PROP_MAP.get(name)
            auto_prop_id = getattr(cv2, auto_prop_name, None) if auto_prop_name else None
            min_v, max_v = _range_for_control(name, backend)
            step = CONTROL_DEFAULT_STEPS.get(name, 1.0)
            entry = {
                "supported": False,
                "readable": False,
                "writable": False,
                "value": None,
                "min": min_v,
                "max": max_v,
                "step": step,
                "default": min_v,
                "unit": CONTROL_UNITS.get(name),
                "auto_supported": auto_prop_id is not None,
                "auto_value": None,
                "backend_property": prop_name,
                "range_source": "default_profile",
            }
            if prop_id is None:
                entry["warning"] = "property_not_exposed_by_opencv"
                unsupported.append(name)
                controls[name] = entry
                continue
            try:
                value = float(cap.get(prop_id))
                raw_get[name] = value
                supported = not (value != value) and value not in (-1e9, 1e9)
                entry["supported"] = bool(supported)
                entry["readable"] = bool(supported)
                entry["writable"] = bool(supported)
                entry["value"] = value if supported else None
            except Exception:
                entry["warning"] = "read_failed"
                unsupported.append(name)
                controls[name] = entry
                continue
            if auto_prop_id is not None:
                try:
                    auto_raw = float(cap.get(auto_prop_id))
                    entry["auto_value"] = auto_raw >= 1.0 or abs(auto_raw - 0.75) < 1e-3
                    raw_get[f"{name}_auto"] = auto_raw
                except Exception:
                    entry["auto_value"] = None
                    entry["warning"] = "auto_read_failed"
            controls[name] = entry
        return {
            "camera_index": camera_index,
            "backend": backend,
            "controls": controls,
            "diagnostics": {
                "raw_get": raw_get,
                "unsupported_properties": unsupported,
                "last_apply_errors": [],
            },
        }
    finally:
        cap.release()


def apply_camera_controls(camera_index: int, updates: dict[str, Any], cv2_module: Any | None = None) -> dict[str, Any]:
    cv2 = _cv2(cv2_module)
    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise RuntimeError(f"Camera index {camera_index} is unavailable.")
    try:
        applied: dict[str, Any] = {}
        errors: list[str] = []
        for name, payload in updates.items():
            if name not in CONTROL_PROP_MAP:
                applied[name] = {"requested": payload, "applied": False, "warnings": ["unsupported_property"]}
                continue
            if not isinstance(payload, dict):
                applied[name] = {"requested": payload, "applied": False, "warnings": ["invalid_payload"]}
                continue
            prop_id = getattr(cv2, CONTROL_PROP_MAP[name], None)
            auto_prop_name = CONTROL_AUTO_PROP_MAP.get(name)
            auto_prop_id = getattr(cv2, auto_prop_name, None) if auto_prop_name else None
            item_result = {"requested": payload, "applied": False, "warnings": []}
            if prop_id is None:
                item_result["warnings"].append("unsupported_property")
                errors.append(f"{name}:unsupported_property")
                applied[name] = item_result
                continue
            if "auto" in payload and auto_prop_id is not None:
                try:
                    auto_value = payload.get("auto")
                    if auto_value is not None:
                        # AVFoundation/others can use 0/1; some use 0.25/0.75 semantics.
                        cap.set(auto_prop_id, 1.0 if bool(auto_value) else 0.0)
                except Exception:
                    item_result["warnings"].append("auto_apply_failed")
                    errors.append(f"{name}:auto_apply_failed")
            if "value" in payload and payload.get("value") is not None:
                try:
                    ok = bool(cap.set(prop_id, float(payload["value"])))
                    item_result["applied"] = ok
                    if not ok:
                        item_result["warnings"].append("set_returned_false")
                        errors.append(f"{name}:set_returned_false")
                except Exception:
                    item_result["warnings"].append("value_apply_failed")
                    errors.append(f"{name}:value_apply_failed")
            applied[name] = item_result
        snapshot = camera_controls_snapshot(camera_index, cv2_module=cv2)
        snapshot["applied"] = applied
        diagnostics = snapshot.get("diagnostics") if isinstance(snapshot.get("diagnostics"), dict) else {}
        diagnostics["last_apply_errors"] = errors
        snapshot["diagnostics"] = diagnostics
        return snapshot
    finally:
        cap.release()


def _range_for_control(name: str, backend: str | None) -> tuple[float | None, float | None]:
    if name == "exposure":
        if (backend or "").upper() == "AVFOUNDATION":
            return (-13.0, 0.0)
        return (-20.0, 20.0)
    return CONTROL_DEFAULT_RANGES.get(name, (None, None))
