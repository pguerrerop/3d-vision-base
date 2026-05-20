from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

import cv2

from vision_3d_acquisition.acquisition.preview import PreviewWriter, preview_image_path, read_preview_metadata
from vision_3d_acquisition.acquisition.usb_camera import capture_single_frame
from vision_3d_acquisition.api.filesystem import get_take_detail, list_takes, read_json
from vision_3d_acquisition.api.settings import ApiSettings, get_settings
from vision_3d_acquisition.calibration.calibration_models import (
    Camera2DBeltPlane,
    Camera2DIntrinsics,
    Camera2DTargetConfig,
    ObjectFilterConfig,
    PlaneCalibration,
    PlaneLabel,
    SystemCalibration,
)
from vision_3d_acquisition.calibration.camera_2d import aruco_runtime_diagnostics, calibrate_from_detections, detect_target_in_image
from vision_3d_acquisition.calibration.charuco import generate_charuco_board, save_charuco_pdf, save_charuco_png
from vision_3d_acquisition.calibration.compatibility import evaluate_calibration_compatibility, pick_recommended_calibration
from vision_3d_acquisition.calibration.calibration_storage import (
    load_calibration,
    list_calibrations,
    save_calibration,
)
from vision_3d_acquisition.calibration.plane_detection import detect_plane_candidates, public_plane_candidate
from vision_3d_acquisition.calibration.visualization import generate_plane_visualizations
from vision_3d_acquisition.contracts.modalities import infer_modalities
from vision_3d_acquisition.processing.pointcloud_io import load_point_cloud
from vision_3d_acquisition.runtime_config import read_runtime_config


router = APIRouter(prefix="/api/calibration", tags=["calibration"])
CAPTURE_FALLBACK_MAX_AGE_SECONDS = 2.0


class DetectPlanesRequest(BaseModel):
    take_id: str


class SaveCalibrationRequest(BaseModel):
    reference_take_id: str
    plane_labels: dict[str, PlaneLabel]
    object_filter: ObjectFilterConfig = ObjectFilterConfig()


class Capture2DFrameRequest(BaseModel):
    source_id: str = "usb_camera_0"


class Refresh2DSourceRequest(BaseModel):
    source_id: str = "usb_camera_0"


class Detect2DCornersRequest(BaseModel):
    capture_ids: list[str]
    target: Camera2DTargetConfig


class Calibrate2DRequest(BaseModel):
    source_id: str
    capture_ids: list[str]
    target: Camera2DTargetConfig
    reference_plane_z_mm: float = 0.0


class Save2DCalibrationRequest(BaseModel):
    calibration_id: str
    source_id: str
    target: Camera2DTargetConfig
    intrinsics: Camera2DIntrinsics
    belt_plane: Camera2DBeltPlane


class GenerateCharucoTargetRequest(BaseModel):
    squares_x: int = 7
    squares_y: int = 5
    square_length_mm: float = 25.0
    marker_length_mm: float = 18.0
    dictionary: str = "DICT_4X4_50"


@router.get("/reference-takes")
def reference_takes(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return [
        take.model_dump(mode="json")
        for take in list_takes(settings)
        if take.has_done and "point_cloud" in take.modalities and _point_cloud_path(settings, take.take_id) is not None
    ]


@router.post("/detect-planes")
def detect_planes(
    request: DetectPlanesRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    point_cloud_path = _point_cloud_path(settings, request.take_id)
    if point_cloud_path is None:
        raise HTTPException(status_code=404, detail="Point cloud not found")

    pcd = load_point_cloud(point_cloud_path)
    detected = detect_plane_candidates(pcd)
    public_candidates = [public_plane_candidate(candidate) for candidate in detected]
    output_dir = settings.data_dir / "calibration" / request.take_id
    files = generate_plane_visualizations(detected, output_dir)
    (output_dir / "plane_candidates.json").write_text(
        _json(public_candidates),
        encoding="utf-8",
    )
    return {
        "take_id": request.take_id,
        "preview_image": _calibration_file_url(request.take_id, files["preview_image"]),
        "planes": [
            {
                **candidate,
                "preview_image": _calibration_file_url(request.take_id, files.get(candidate["plane_id"], "")),
            }
            for candidate in public_candidates
        ],
    }


@router.post("/save")
def save(request: SaveCalibrationRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, str]:
    candidate_path = settings.data_dir / "calibration" / request.reference_take_id / "plane_candidates.json"
    if not candidate_path.is_file():
        raise HTTPException(status_code=404, detail="Detect plane candidates before saving calibration")
    candidates = read_json(candidate_path)
    if not isinstance(candidates, list):
        import json

        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    planes: list[PlaneCalibration] = []
    for candidate in candidates:
        plane_id = str(candidate["plane_id"])
        label = request.plane_labels.get(plane_id, "unused")
        planes.append(
            PlaneCalibration(
                plane_id=plane_id,
                label=label,
                plane_model=tuple(candidate["plane_model"]),
                point_count=int(candidate["point_count"]),
                bbox_min_mm=tuple(candidate["bbox_min_mm"]),
                bbox_max_mm=tuple(candidate["bbox_max_mm"]),
                extent_mm=tuple(candidate["extent_mm"]),
                avg_z_mm=float(candidate["avg_z_mm"]),
                normal=tuple(candidate["normal"]) if "normal" in candidate else None,
                roi_polygon_xy_mm=[tuple(point) for point in candidate.get("roi_polygon_xy_mm", [])],
            )
        )
    try:
        calibration = SystemCalibration(
            calibration_id=_calibration_id(request.reference_take_id),
            calibration_type="plane_3d",
            source_modalities=["point_cloud"],
            created_at=datetime.now(UTC),
            reference_take_id=request.reference_take_id,
            planes=planes,
            object_filter=request.object_filter,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = save_calibration(calibration)
    return {"calibration_id": path.stem, "path": str(path)}


@router.post("/camera-2d/capture")
def capture_2d_frame(request: Capture2DFrameRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    captured_at = datetime.now(UTC)
    acquired = _acquire_capture_frame(request.source_id, settings)
    if str(acquired.get("freshness_status")) != "fresh":
        age = acquired.get("frame_age_seconds")
        _raise_capture_error(
            "NO_FRESH_FRAME_AVAILABLE",
            f"No fresh frame available for {request.source_id}",
            status_code=409,
            frame_age_seconds=age,
        )
    frame = acquired["frame"]
    capture_id = f"capture_{captured_at.strftime('%Y%m%d_%H%M%S_%f')}_{request.source_id}"
    capture_image_path = _captures_dir(settings) / f"{capture_id}.png"
    if not cv2.imwrite(str(capture_image_path), frame):
        _raise_capture_error("FRAME_SAVE_FAILED", "Failed to persist captured calibration frame", status_code=500)
    image_height, image_width = frame.shape[:2]
    capture_record = {
        "id": capture_id,
        "source_id": request.source_id,
        "timestamp": captured_at.isoformat(),
        "image_path": str(capture_image_path),
        "image_url": _capture_image_url(capture_id),
        "resolution": [int(image_width), int(image_height)],
        "stale": False,
        "detected_corners": None,
        "target_type": None,
    }
    _capture_record_path(settings, capture_id).write_text(_json(capture_record), encoding="utf-8")
    _update_preview_from_capture(settings, request.source_id, frame)
    return {
        "capture_id": capture_id,
        "image_path": str(capture_image_path),
        "timestamp": capture_record["timestamp"],
        "resolution": capture_record["resolution"],
        "source_id": request.source_id,
        "acquisition_mode": acquired["acquisition_mode"],
        "frame_timestamp": acquired["frame_timestamp"],
        "frame_age_seconds": acquired["frame_age_seconds"],
        "freshness_status": acquired["freshness_status"],
        "fallback_used": bool(acquired.get("fallback_used", False)),
        "camera_index": acquired.get("camera_index"),
        "backend": acquired.get("backend"),
        "direct_capture_success": bool(acquired.get("direct_capture_success", False)),
        "image_url": _capture_image_url(capture_id),
    }


@router.get("/camera-2d/captures")
def list_2d_captures(source_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    captures = [_public_capture_record(item) for item in _list_capture_records(settings, source_id=source_id)]
    return {"captures": captures}


@router.get("/camera-2d/captures/{capture_id}/image")
def capture_2d_image(capture_id: str, settings: ApiSettings = Depends(get_settings)) -> FileResponse:
    record = _load_capture_record(settings, capture_id)
    image_path = Path(str(record.get("image_path") or ""))
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail={"code": "NO_PREVIEW_FRAME_AVAILABLE", "message": f"Capture image missing: {capture_id}"})
    base = _captures_dir(settings).resolve()
    resolved = image_path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail={"code": "SOURCE_NOT_AVAILABLE", "message": "Capture image path is outside calibration captures directory"}) from exc
    suffix = resolved.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else None
    return FileResponse(resolved, media_type=media_type)


@router.post("/camera-2d/refresh-source")
def refresh_2d_source(request: Refresh2DSourceRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    frame = _acquire_fresh_source_frame(request.source_id, settings)
    meta = _update_preview_from_capture(settings, request.source_id, frame)
    return {"source_id": request.source_id, "status": "live", "metadata": meta}


@router.post("/camera-2d/detect-corners")
def detect_2d_corners(request: Detect2DCornersRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not request.capture_ids:
        raise HTTPException(status_code=400, detail="capture_ids must not be empty")
    detections: list[dict[str, Any]] = []
    for capture_id in request.capture_ids:
        record = _load_capture_record(settings, capture_id)
        image_path = str(record.get("image_path") or "")
        if not image_path:
            detections.append({"capture_id": capture_id, "corners_found": False, "corner_count": 0, "error": "missing image_path"})
            continue
        try:
            detection = detect_target_in_image(Path(image_path), request.target)
        except Exception as exc:
            detections.append({"capture_id": capture_id, "image_path": image_path, "corners_found": False, "corner_count": 0, "error": str(exc)})
            continue
        debug_path = Path(image_path).with_name(f"{Path(image_path).stem}_corners.png")
        cv2.imwrite(str(debug_path), detection.debug_image)
        record["detected_corners"] = {
            "corners_found": detection.corners_found,
            "corner_count": detection.corner_count,
            "debug_image_path": str(debug_path),
            "detected_at": datetime.now(UTC).isoformat(),
        }
        record["target_type"] = request.target.type
        _capture_record_path(settings, capture_id).write_text(_json(record), encoding="utf-8")
        detections.append(
            {
                "capture_id": capture_id,
                "image_path": image_path,
                "success": detection.success,
                "error_code": detection.error_code,
                "corners_found": detection.corners_found,
                "corner_count": detection.corner_count,
                "marker_count": detection.marker_count,
                "charuco_corner_count": detection.corner_count,
                "api_mode": detection.api_mode,
                "image_width": detection.image_width,
                "image_height": detection.image_height,
                "debug_image_path": str(debug_path),
                "warnings": detection.warnings,
            }
        )
    found = sum(1 for item in detections if item.get("corners_found"))
    if found == 0:
        error_code = "NO_CORNERS_DETECTED"
        if any(item.get("error_code") == "CHARUCO_API_UNAVAILABLE" for item in detections):
            error_code = "CHARUCO_API_UNAVAILABLE"
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": error_code,
                "detections": detections,
                "capture_count": len(detections),
                "detected_count": 0,
            },
        )
    return {"success": True, "detections": detections, "capture_count": len(detections), "detected_count": found}


@router.get("/camera-2d/aruco-diagnostics")
def camera_2d_aruco_diagnostics() -> dict[str, Any]:
    return aruco_runtime_diagnostics()


@router.post("/camera-2d/calibrate")
def calibrate_2d(request: Calibrate2DRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not request.capture_ids:
        raise HTTPException(status_code=400, detail="capture_ids must not be empty")
    image_paths: list[str] = []
    for capture_id in request.capture_ids:
        record = _load_capture_record(settings, capture_id)
        image_path = str(record.get("image_path") or "")
        if not image_path:
            raise HTTPException(status_code=400, detail=f"Capture {capture_id} is missing image_path")
        image_paths.append(image_path)
    frames = [detect_target_in_image(Path(image_path), request.target) for image_path in image_paths]
    result = calibrate_from_detections(frames)
    return {
        "source_id": request.source_id,
        "target": request.target.model_dump(mode="json"),
        "intrinsics": {
            "camera_matrix": result["camera_matrix"],
            "dist_coeffs": result["dist_coeffs"],
            "reprojection_error": result["reprojection_error"],
            "image_width": result["image_width"],
            "image_height": result["image_height"],
        },
        "belt_plane": {
            "homography": result["homography"],
            "mm_per_px_x": result["mm_per_px_x"],
            "mm_per_px_y": result["mm_per_px_y"],
            "reference_plane_z_mm": request.reference_plane_z_mm,
        },
        "diagnostics": {
            "used_frames": result["used_frames"],
            "corner_residuals": result["residuals"],
            "homography_inliers": result["inlier_count"],
        },
    }


@router.post("/camera-2d/save")
def save_2d_calibration(request: Save2DCalibrationRequest) -> dict[str, str]:
    calibration = SystemCalibration(
        calibration_id=request.calibration_id,
        calibration_type="camera_2d",
        source_modalities=["rgb"],
        source_id=request.source_id,
        target=request.target,
        intrinsics=request.intrinsics,
        belt_plane=request.belt_plane,
        created_at=datetime.now(UTC),
    )
    path = save_calibration(calibration)
    return {"calibration_id": path.stem, "path": str(path)}


@router.post("/camera-2d/generate-charuco")
def generate_charuco_target(
    request: GenerateCharucoTargetRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, str]:
    output_dir = settings.data_dir / "calibration" / "targets"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_{request.squares_x}x{request.squares_y}_{int(round(request.square_length_mm))}mm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    image = generate_charuco_board(
        squares_x=request.squares_x,
        squares_y=request.squares_y,
        square_length_mm=request.square_length_mm,
        marker_length_mm=request.marker_length_mm,
        dictionary=request.dictionary,
    )
    save_charuco_png(png_path, image)
    save_charuco_pdf(pdf_path, image)
    return {"png_path": str(png_path), "pdf_path": str(pdf_path)}


@router.get("/list")
def list_saved_calibrations() -> list[dict[str, Any]]:
    return [calibration.model_dump(mode="json") for calibration in list_calibrations()]


@router.get("/active")
def active_calibration(take_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    status = read_runtime_config()
    take_modalities: list[str] | None = None
    metadata: dict[str, Any] | None = None
    if take_id:
        detail = get_take_detail(settings, take_id)
        take_modalities = detail.modalities if detail is not None else []
        metadata = detail.metadata if detail is not None else None
    configured = Path(status.default_calibration_file) if status.default_calibration_file else None
    if configured is None:
        return _active_payload(False, None, None, "none", status.warning, take_modalities=take_modalities)
    if not status.default_calibration_valid:
        return _active_payload(False, str(configured), None, "missing", status.warning, take_modalities=take_modalities)
    try:
        calibration = load_calibration(configured)
    except FileNotFoundError:
        return _active_payload(False, str(configured), None, "missing", status.warning, take_modalities=take_modalities)
    valid_for_take = True
    compatibility_status = "valid"
    warning = None
    diagnostics: dict[str, Any] | None = None
    if take_modalities is not None:
        diagnostics = evaluate_calibration_compatibility(
            calibration,
            take_modalities=take_modalities,
            metadata=metadata if isinstance(metadata, dict) else None,
        )
        missing = diagnostics["missing_modalities"]
        if missing:
            valid_for_take = False
            compatibility_status = "incompatible"
            warning = f"Calibration requires {', '.join(missing)} but take has {', '.join(take_modalities) or 'none'}"
        elif diagnostics["status"] == "warning":
            compatibility_status = "warning"
            warning = "; ".join(diagnostics["warnings"]) if diagnostics["warnings"] else None
    return _active_payload(
        True,
        str(configured),
        calibration.calibration_id,
        compatibility_status,
        warning,
        calibration=calibration,
        take_modalities=take_modalities,
        valid_for_take=valid_for_take,
        diagnostics=diagnostics,
    )


@router.get("/recommended")
def recommended_calibration(take_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    take_modalities: list[str] | None = None
    sensor_setup_hint: str | None = None
    if take_id:
        detail = get_take_detail(settings, take_id)
        if detail is not None:
            take_modalities = detail.modalities
            metadata = detail.metadata if isinstance(detail.metadata, dict) else {}
            sensor = metadata.get("sensor") if isinstance(metadata.get("sensor"), dict) else {}
            sensor_setup_hint = str(sensor.get("model") or "")
    calibrations = list_calibrations()
    recommended = pick_recommended_calibration(
        calibrations,
        take_modalities=take_modalities,
        sensor_setup_hint=sensor_setup_hint,
    )
    if recommended is None:
        return {"recommended": None}
    return {"recommended": recommended.model_dump(mode="json")}


@router.get("/files/{take_id}/{filename}")
def calibration_file(take_id: str, filename: str, settings: ApiSettings = Depends(get_settings)) -> FileResponse:
    if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="File not found")
    base = (settings.data_dir / "calibration" / take_id).resolve()
    path = (base / filename).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@router.get("/{calibration_id}")
def get_calibration(calibration_id: str) -> dict[str, Any]:
    try:
        return load_calibration(calibration_id).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Calibration not found") from exc


def _point_cloud_path(settings: ApiSettings, take_id: str) -> Path | None:
    metadata = read_json(settings.incoming_dir / take_id / "metadata.json") or {}
    if "point_cloud" not in infer_modalities(metadata, settings.incoming_dir / take_id):
        return None
    filename = metadata.get("files", {}).get("point_cloud", "point_cloud.ply")
    path = settings.incoming_dir / take_id / str(filename)
    return path if path.is_file() else None


def _calibration_file_url(take_id: str, filename: str) -> str:
    return f"/api/calibration/files/{take_id}/{filename}"


def _calibration_id(take_id: str) -> str:
    return f"belt_setup_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}_{take_id}"


def _json(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2) + "\n"


def _captures_dir(settings: ApiSettings) -> Path:
    path = settings.data_dir / "calibration" / "captures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _capture_record_path(settings: ApiSettings, capture_id: str) -> Path:
    return _captures_dir(settings) / f"{capture_id}.json"


def _load_capture_record(settings: ApiSettings, capture_id: str) -> dict[str, Any]:
    path = _capture_record_path(settings, capture_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Capture not found: {capture_id}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid capture record: {capture_id}")
    return payload


def _list_capture_records(settings: ApiSettings, source_id: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(_captures_dir(settings).glob("*.json"), reverse=True):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        if source_id and str(payload.get("source_id") or "") != source_id:
            continue
        records.append(payload)
    return records


def _public_capture_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    capture_id = str(payload.get("id") or "")
    payload["image_url"] = str(payload.get("image_url") or _capture_image_url(capture_id))
    return payload


def _capture_image_url(capture_id: str) -> str:
    return f"/api/calibration/camera-2d/captures/{capture_id}/image"


def _acquire_fresh_source_frame(source_id: str, settings: ApiSettings) -> Any:
    if source_id.startswith("usb_camera_"):
        try:
            camera_index = int(source_id.rsplit("_", 1)[1])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid source id: {source_id}") from exc
        cap = cv2.VideoCapture(camera_index)
        if not cap or not cap.isOpened():
            raise HTTPException(status_code=404, detail=f"Source unavailable: {source_id}")
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail=f"Failed to acquire frame from {source_id}")
        return frame

    source_path = preview_image_path(settings.data_dir, source_id)
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"Source unavailable: {source_id}")
    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode source frame: {source_id}")
    return frame


def _acquire_capture_frame(source_id: str, settings: ApiSettings) -> dict[str, Any]:
    direct_error: dict[str, Any] | None = None

    if source_id.startswith("usb_camera_"):
        camera_index = _camera_index_from_source_id(source_id)
        try:
            direct = capture_single_frame(data_dir=settings.data_dir, camera_index=camera_index, session_id=None)
            frame_age = _frame_age_seconds(direct.timestamp) or 0.0
            return {
                "frame": direct.frame,
                "acquisition_mode": "direct_usb_camera",
                "frame_timestamp": direct.timestamp,
                "frame_age_seconds": frame_age,
                "freshness_status": "fresh",
                "fallback_used": False,
                "camera_index": camera_index,
                "backend": direct.backend,
                "direct_capture_success": True,
            }
        except Exception as exc:
            direct_error = {"code": "CAMERA_BUSY", "message": f"Direct USB capture failed for {source_id}: {exc}"}

        refreshed_cached = _refresh_preview_cache(source_id, settings, preview_owns_camera=False)
        cached = refreshed_cached if refreshed_cached is not None else _acquire_from_preview_cache(source_id, settings)
        if cached is None:
            if direct_error is not None:
                _raise_capture_error(direct_error["code"], direct_error["message"], status_code=503)
            _raise_capture_error("NO_PREVIEW_FRAME_AVAILABLE", f"No preview frame available for {source_id}", status_code=404)
        cached["acquisition_mode"] = "preview_cache_fallback"
        cached["fallback_used"] = True
        cached["freshness_status"] = "fresh" if _is_cache_fresh(cached.get("frame_age_seconds")) else "stale"
        cached["camera_index"] = camera_index
        cached["backend"] = "preview_cache"
        cached["direct_capture_success"] = False
        return cached

    cached = _acquire_from_preview_cache(source_id, settings)
    if cached is None:
        _raise_capture_error("NO_PREVIEW_FRAME_AVAILABLE", f"No preview frame available for {source_id}", status_code=404)
    cached["acquisition_mode"] = "preview_cache_fallback"
    cached["fallback_used"] = True
    cached["freshness_status"] = "fresh" if _is_cache_fresh(cached.get("frame_age_seconds")) else "stale"
    return cached


def _acquire_from_preview_cache(source_id: str, settings: ApiSettings) -> dict[str, Any] | None:
    source_path = preview_image_path(settings.data_dir, source_id)
    if not source_path.is_file():
        return None
    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    metadata = read_preview_metadata(settings.data_dir, source_id, stale_seconds=10_000)
    frame_timestamp = metadata.get("timestamp")
    return {
        "frame": frame,
        "frame_timestamp": frame_timestamp,
        "frame_age_seconds": _frame_age_seconds(frame_timestamp),
    }


def _refresh_preview_cache(source_id: str, settings: ApiSettings, *, preview_owns_camera: bool) -> dict[str, Any] | None:
    if source_id.startswith("usb_camera_") and not preview_owns_camera:
        try:
            frame = _acquire_fresh_source_frame(source_id, settings)
            _update_preview_from_capture(settings, source_id, frame)
        except HTTPException:
            return None
    return _acquire_from_preview_cache(source_id, settings)


def _is_cache_fresh(frame_age_seconds: Any) -> bool:
    if frame_age_seconds is None:
        return False
    try:
        age = float(frame_age_seconds)
    except Exception:
        return False
    return age <= CAPTURE_FALLBACK_MAX_AGE_SECONDS


def _camera_index_from_source_id(source_id: str) -> int:
    try:
        return int(source_id.rsplit("_", 1)[1])
    except Exception:
        _raise_capture_error("SOURCE_NOT_AVAILABLE", f"Invalid source id: {source_id}", status_code=400)
        return 0


def _raise_capture_error(code: str, message: str, *, status_code: int, frame_age_seconds: float | None = None) -> None:
    detail: dict[str, Any] = {"code": code, "message": message}
    if frame_age_seconds is not None:
        detail["frame_age_seconds"] = float(frame_age_seconds)
    raise HTTPException(status_code=status_code, detail=detail)


def _update_preview_from_capture(settings: ApiSettings, source_id: str, frame: Any) -> dict[str, Any]:
    if source_id.startswith("usb_camera_"):
        camera_index = int(source_id.rsplit("_", 1)[1])
        writer = PreviewWriter(settings.data_dir, source="usb_camera", camera_index=camera_index, interval_ms=1, cv2_module=cv2)
        return writer.export(frame).model_dump()
    out_path = preview_image_path(settings.data_dir, source_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return read_preview_metadata(settings.data_dir, source_id)


def _frame_age_seconds(timestamp: Any) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())


def _active_payload(
    active: bool,
    path: str | None,
    calibration_id: str | None,
    status: str,
    warning: str | None,
    *,
    calibration: SystemCalibration | None = None,
    take_modalities: list[str] | None = None,
    valid_for_take: bool | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calibration_type = calibration.calibration_type if calibration is not None else "plane_3d"
    source_modalities = list(calibration.source_modalities) if calibration is not None else ["point_cloud"]
    return {
        "active": active,
        "path": path,
        "calibration_id": calibration_id,
        "status": status,
        "warning": warning,
        "calibration_type": calibration_type,
        "source_modalities": source_modalities,
        "required_modalities": source_modalities,
        "take_modalities": take_modalities,
        "valid_for_take": valid_for_take,
        "compatibility": diagnostics,
    }
