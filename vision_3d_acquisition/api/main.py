from __future__ import annotations

import csv
import io
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from vision_3d_acquisition.acquisition.preview import preview_image_path, read_preview_metadata
from vision_3d_acquisition.acquisition.usb_camera import capture_image, discover_cameras, record_video
from vision_3d_acquisition.api.calibration import router as calibration_router
from vision_3d_acquisition.api.events import stream_events
from vision_3d_acquisition.api.filesystem import (
    count_take_dirs,
    get_session_summary,
    get_take_detail,
    list_session_summaries,
    latest_take,
    list_takes,
    read_runtime_state,
    safe_take_file,
)
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.api.histogram import load_or_compute_histogram, resolve_source_image_path
from vision_3d_acquisition.api.schemas import DatasetSessionSummary, DatasetSummary, HealthResponse, RuntimeState, TakeDetail, TakeSummary
from vision_3d_acquisition.api.settings import ApiSettings, get_cors_origins, get_settings
from vision_3d_acquisition.poc.exports import export_labeled_dataset_summary, export_object_metrics
from vision_3d_acquisition.poc.labels import ALLOWED_LABELS, list_labeled_takes, load_labels, save_labels
from vision_3d_acquisition.poc.summary import build_poc_run_summary, validate_result_payload
from vision_3d_acquisition.pipelines.registry import list_pipelines
from vision_3d_acquisition.apps.ball_inspection import run_ball_inspection_flow
from vision_3d_acquisition.processing.pipeline import load_processing_config, process_take
from vision_3d_acquisition.processes import ProcessService
from vision_3d_acquisition.runtime_config import (
    RuntimeConfig,
    clear_default_calibration,
    read_runtime_config,
    set_default_calibration,
    warn_if_deprecated_processing_local_calibration,
    write_runtime_config,
)


app = FastAPI(title="3D Acquisition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_cors_origins()),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calibration_router)


@app.on_event("startup")
def _warn_deprecated_processing_local_calibration() -> None:
    warn_if_deprecated_processing_local_calibration()


class LabelRequest(BaseModel):
    labels: list[str]
    notes: str | None = None
    reviewer: str | None = None


class DefaultCalibrationRequest(BaseModel):
    path: str


class UsbCaptureRequest(BaseModel):
    camera_index: int = 0
    session: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    preview_interval_ms: int = 250
    output_name: str | None = None


class UsbVideoCaptureRequest(UsbCaptureRequest):
    duration: float = 10.0


class CreatePipelineInstanceRequest(BaseModel):
    name: str
    template_id: str
    input_type: str


class UpdatePipelineInstanceRequest(BaseModel):
    configured_steps: list[dict[str, Any]] | None = None
    overrides: dict[str, Any] | None = None


class ExecutePipelineInstanceRequest(BaseModel):
    image_path: str


class PromoteRecipeRequest(BaseModel):
    run_id: str | None = None
    recipe_type: str = "poc_baseline"
    notes: str | None = None


class ExecuteTakeRequest(BaseModel):
    pipeline_id: str = "3d_ball_inspection"
    run_until_stage: str | None = None
    reprocess: bool = False


class DatasetRequest(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    tags: list[str] = []
    notes: str | None = None


class DatasetSessionRequest(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    calibration_id: str | None = None
    sensor_metadata: dict[str, Any] = {}
    conveyor_metadata: dict[str, Any] = {}
    lighting_metadata: dict[str, Any] = {}
    tags: list[str] = []
    notes: str | None = None


class TakeMetadataUpdateRequest(BaseModel):
    friendly_name: str | None = None
    labels: list[str] | None = None
    tags: list[str] | None = None
    notes: str | None = None
    expected_class: str | None = None
    expected_diameter_mm: float | None = None
    expected_count: int | None = None
    operator_notes: str | None = None
    validation_status: str | None = None
    dataset_id: str | None = None
    session_id: str | None = None


class ObjectAnnotationUpsertRequest(BaseModel):
    id: str | None = None
    source_stage: str
    source_artifact_id: str | None = None
    candidate_id: str
    bbox: list[float] | None = None
    centroid: list[float] | None = None
    contour_ref: str | None = None
    labels: list[str] = []
    expected_class: str | None = None
    expected_diameter_mm: float | None = None
    notes: str | None = None
    validation_status: str = "unreviewed"


class CaptureTakeRequest(BaseModel):
    dataset_id: str
    dataset_session_id: str
    friendly_name: str | None = None
    tags: list[str] = []
    expected_class: str | None = None
    expected_diameter_mm: float | None = None
    notes: str | None = None
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    preview_interval_ms: int = 250


PROCESS_SERVICE_PIPELINE_TO_TEMPLATE: dict[str, str] = {
    "mining_steel_ball_classification_2d": "mining_steel_ball_classification_2d_reflectance_mvp",
}


@app.get("/api/health", response_model=HealthResponse)
def health(settings: ApiSettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        data_dir=str(settings.data_dir),
        incoming_count=count_take_dirs(settings.incoming_dir),
        processed_count=count_take_dirs(settings.processed_dir),
    )


@app.get("/api/state", response_model=RuntimeState)
def state(settings: ApiSettings = Depends(get_settings)) -> RuntimeState:
    return read_runtime_state(settings)


@app.get("/api/pipelines")
def pipelines() -> list[dict[str, Any]]:
    return list_pipelines()


@app.get("/api/process-templates")
def process_templates(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return ProcessService(settings).list_templates()


@app.get("/api/pipeline-instances")
def pipeline_instances(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return ProcessService(settings).list_instances()


@app.post("/api/pipeline-instances")
def create_pipeline_instance(payload: CreatePipelineInstanceRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ProcessService(settings).create_instance(
            name=payload.name,
            template_id=payload.template_id,
            supported_input_type=payload.input_type,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/pipeline-instances/{instance_id}")
def update_pipeline_instance(instance_id: str, payload: UpdatePipelineInstanceRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ProcessService(settings).update_instance(
            instance_id,
            configured_steps=payload.configured_steps,
            overrides=payload.overrides,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pipeline-instances/{instance_id}/execute")
def execute_pipeline_instance(instance_id: str, payload: ExecutePipelineInstanceRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ProcessService(settings).execute(instance_id, image_path=settings.data_dir / payload.image_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipeline-instances/{instance_id}/recipes")
def promote_pipeline_recipe(instance_id: str, payload: PromoteRecipeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ProcessService(settings).promote_recipe(
            instance_id,
            run_id=payload.run_id,
            recipe_type=payload.recipe_type,
            notes=payload.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/cameras")
def cameras(max_index: int = 8) -> list[dict[str, Any]]:
    return [camera.model_dump() for camera in discover_cameras(max_index=max_index)]


@app.get("/api/sources")
def sources(settings: ApiSettings = Depends(get_settings), max_index: int = 1, stale_after_seconds: float = 5.0) -> list[dict[str, Any]]:
    runtime = read_runtime_state(settings)
    by_id: dict[str, dict[str, Any]] = {}

    for camera in discover_cameras(max_index=max_index):
        source_id = f"usb_camera_{camera.index}"
        by_id[source_id] = {
            "id": source_id,
            "label": f"USB Camera {camera.index}",
            "type": "usb_camera",
            "modality": "rgb",
            "status": "unavailable",
            "last_frame_at": None,
            "last_frame_age_seconds": None,
            "resolution": camera.resolution(),
            "fps": None,
            "active": runtime.acquisition_source == "usb_camera" and int((runtime.acquisition_source_details or {}).get("camera_index", -1)) == camera.index,
            "session_id": runtime.current_session,
        }
        if camera.working:
            by_id[source_id]["status"] = "stale"

    preview_dir = settings.data_dir / "runtime" / "previews"
    if preview_dir.is_dir():
        for metadata_path in preview_dir.glob("*.json"):
            source_id = metadata_path.stem
            metadata = read_preview_metadata(settings.data_dir, source_id, stale_seconds=stale_after_seconds)
            timestamp = str(metadata.get("timestamp") or "") or None
            age = _age_seconds(timestamp)
            if timestamp is None:
                status = "unavailable"
            elif age is not None and age <= stale_after_seconds:
                status = "live"
            else:
                status = "stale"
            existing = by_id.get(source_id, {})
            if not existing:
                source_type = "simulated"
                if source_id.startswith("usb_camera_"):
                    source_type = "usb_camera"
                elif source_id.startswith("sick_"):
                    source_type = "sick_3d"
                existing = {
                    "id": source_id,
                    "label": source_id.replace("_", " ").title(),
                    "type": source_type,
                    "modality": "rgb",
                    "active": runtime.preview_source == source_id,
                    "session_id": runtime.current_session,
                }
            existing.update(
                {
                    "status": status,
                    "last_frame_at": timestamp,
                    "last_frame_age_seconds": age,
                    "resolution": metadata.get("resolution") if isinstance(metadata.get("resolution"), list) else existing.get("resolution"),
                    "fps": metadata.get("fps_estimate"),
                }
            )
            by_id[source_id] = existing

    return sorted(by_id.values(), key=lambda item: item.get("id") or "")


@app.post("/api/capture/image")
def capture_rgb_image(payload: UsbCaptureRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        take_id, folder = capture_image(
            data_dir=settings.data_dir,
            camera_index=payload.camera_index,
            session_id=payload.session,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            preview_window=False,
            preview_interval_ms=payload.preview_interval_ms,
            output_name=payload.output_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"take_id": take_id, "folder": str(folder), "modalities": ["rgb"]}


@app.post("/api/takes/capture")
def capture_take_to_dataset(payload: CaptureTakeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = DatasetService(settings.data_dir)
    if service.get_dataset(payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if service.get_session(payload.dataset_id, payload.dataset_session_id) is None:
        raise HTTPException(status_code=404, detail="Dataset session not found")
    try:
        take_id, folder = capture_image(
            data_dir=settings.data_dir,
            camera_index=payload.camera_index,
            session_id=payload.dataset_session_id,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            preview_window=False,
            preview_interval_ms=payload.preview_interval_ms,
            output_name=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sidecar = service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=payload.dataset_id,
        session_id=payload.dataset_session_id,
        updates={
            "friendly_name": payload.friendly_name or take_id,
            "tags": payload.tags,
            "expected_class": payload.expected_class,
            "expected_diameter_mm": payload.expected_diameter_mm,
            "notes": payload.notes,
            "validation_status": "unreviewed",
        },
        source_metadata={"session_id": payload.dataset_session_id},
    )
    detail = get_take_detail(settings, take_id)
    return {
        "ok": True,
        "take_id": take_id,
        "folder": str(folder),
        "modalities": ["rgb"],
        "metadata": sidecar,
        "take": detail.model_dump(mode="json") if detail else None,
    }


@app.post("/api/capture/video")
def capture_rgb_video(payload: UsbVideoCaptureRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        take_id, folder = record_video(
            data_dir=settings.data_dir,
            camera_index=payload.camera_index,
            duration=payload.duration,
            session_id=payload.session,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            preview_window=False,
            preview_interval_ms=payload.preview_interval_ms,
            output_name=payload.output_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"take_id": take_id, "folder": str(folder), "modalities": ["rgb", "rgb_video"]}


@app.get("/api/runtime/preview")
def runtime_preview(settings: ApiSettings = Depends(get_settings)) -> FileResponse:
    path = preview_image_path(settings.data_dir)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preview not available")
    return FileResponse(path, media_type="image/jpeg", headers=_preview_headers())


@app.get("/api/runtime/preview/metadata")
def runtime_preview_metadata(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return read_preview_metadata(settings.data_dir)


@app.get("/api/runtime/preview/{source_id}")
def runtime_preview_for_source(source_id: str, settings: ApiSettings = Depends(get_settings)) -> FileResponse:
    path = preview_image_path(settings.data_dir, source_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preview not available")
    return FileResponse(path, media_type="image/jpeg", headers=_preview_headers())


@app.get("/api/runtime-config")
def runtime_config() -> dict[str, Any]:
    return _runtime_config_payload()


@app.put("/api/runtime-config")
def update_runtime_config(payload: RuntimeConfig) -> dict[str, Any]:
    write_runtime_config(payload)
    return _runtime_config_payload()


@app.post("/api/runtime-config/default-calibration")
def update_default_calibration(payload: DefaultCalibrationRequest) -> dict[str, Any]:
    try:
        set_default_calibration(payload.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _runtime_config_payload()


@app.delete("/api/runtime-config/default-calibration")
def delete_default_calibration() -> dict[str, Any]:
    clear_default_calibration()
    return _runtime_config_payload()


@app.get("/api/takes", response_model=list[TakeSummary])
def takes(
    session_id: str | None = None,
    settings: ApiSettings = Depends(get_settings),
    dataset_id: str | None = None,
    validation_status: str | None = None,
    tag: str | None = None,
    search: str | None = None,
) -> list[TakeSummary]:
    return list_takes(settings, session_id=session_id, dataset_id=dataset_id, validation_status=validation_status, tag=tag, search=search)


@app.get("/api/takes/latest", response_model=TakeSummary | None)
def latest(settings: ApiSettings = Depends(get_settings)) -> TakeSummary | None:
    return latest_take(settings)


@app.get("/api/sessions")
def sessions(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return list_session_summaries(settings)


@app.get("/api/datasets", response_model=list[DatasetSummary])
def datasets(settings: ApiSettings = Depends(get_settings)) -> list[DatasetSummary]:
    service = DatasetService(settings.data_dir)
    sessions = service.list_sessions()
    counts: dict[str, dict[str, int]] = {}
    for session in sessions:
        did = str(session.get("dataset_id") or "")
        if not did:
            continue
        counts.setdefault(did, {"session_count": 0, "take_count": 0})
        counts[did]["session_count"] += 1
        folder = settings.data_dir / "datasets" / f"dataset_{did}" / "sessions" / f"session_{session.get('id')}" / "takes"
        take_count = len([child for child in folder.iterdir() if child.is_dir()]) if folder.is_dir() else 0
        counts[did]["take_count"] += take_count
    payload: list[DatasetSummary] = []
    for dataset in service.list_datasets():
        did = str(dataset.get("id") or "")
        info = counts.get(did, {"session_count": 0, "take_count": 0})
        payload.append(DatasetSummary(**dataset, session_count=info["session_count"], take_count=info["take_count"]))
    return payload


@app.post("/api/datasets", response_model=DatasetSummary)
def create_dataset(payload: DatasetRequest, settings: ApiSettings = Depends(get_settings)) -> DatasetSummary:
    service = DatasetService(settings.data_dir)
    created = service.create_dataset(name=payload.name, description=payload.description, tags=payload.tags, notes=payload.notes, dataset_id=payload.id)
    return DatasetSummary(**created, session_count=0, take_count=0)


@app.put("/api/datasets/{dataset_id}", response_model=DatasetSummary)
def update_dataset(dataset_id: str, payload: DatasetRequest, settings: ApiSettings = Depends(get_settings)) -> DatasetSummary:
    service = DatasetService(settings.data_dir)
    updated = service.update_dataset(dataset_id, payload.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetSummary(**updated, session_count=0, take_count=0)


@app.get("/api/datasets/{dataset_id}/sessions", response_model=list[DatasetSessionSummary])
def dataset_sessions(dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> list[DatasetSessionSummary]:
    service = DatasetService(settings.data_dir)
    sessions = service.list_sessions(dataset_id=dataset_id)
    payload: list[DatasetSessionSummary] = []
    for item in sessions:
        sid = str(item.get("id") or "")
        takes_dir = settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "sessions" / f"session_{sid}" / "takes"
        take_count = len([child for child in takes_dir.iterdir() if child.is_dir()]) if takes_dir.is_dir() else 0
        payload.append(DatasetSessionSummary(**item, take_count=take_count))
    return payload


@app.post("/api/datasets/{dataset_id}/sessions", response_model=DatasetSessionSummary)
def create_dataset_session(dataset_id: str, payload: DatasetSessionRequest, settings: ApiSettings = Depends(get_settings)) -> DatasetSessionSummary:
    service = DatasetService(settings.data_dir)
    created = service.create_session(
        dataset_id=dataset_id,
        name=payload.name,
        description=payload.description,
        calibration_id=payload.calibration_id,
        sensor_metadata=payload.sensor_metadata,
        conveyor_metadata=payload.conveyor_metadata,
        lighting_metadata=payload.lighting_metadata,
        tags=payload.tags,
        notes=payload.notes,
        session_id=payload.id,
    )
    return DatasetSessionSummary(**created, take_count=0)


@app.put("/api/datasets/{dataset_id}/sessions/{session_id}", response_model=DatasetSessionSummary)
def update_dataset_session(dataset_id: str, session_id: str, payload: DatasetSessionRequest, settings: ApiSettings = Depends(get_settings)) -> DatasetSessionSummary:
    service = DatasetService(settings.data_dir)
    updated = service.update_session(dataset_id, session_id, payload.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found")
    takes_dir = settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "sessions" / f"session_{session_id}" / "takes"
    take_count = len([child for child in takes_dir.iterdir() if child.is_dir()]) if takes_dir.is_dir() else 0
    return DatasetSessionSummary(**updated, take_count=take_count)


@app.get("/api/sessions/{session_id}/summary")
def session_summary_endpoint(session_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    summary = get_session_summary(settings, session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


@app.get("/api/takes/{take_id}", response_model=TakeDetail)
def take_detail(take_id: str, settings: ApiSettings = Depends(get_settings)) -> TakeDetail:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    return detail


@app.put("/api/takes/{take_id}/metadata")
def update_take_metadata(take_id: str, payload: TakeMetadataUpdateRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    service = DatasetService(settings.data_dir)
    source_meta = detail.metadata if isinstance(detail.metadata, dict) else {}
    current = service.load_take_metadata(take_id=take_id, source_metadata=source_meta)
    dataset_id = payload.dataset_id or str(current.get("dataset_id") or "")
    session_id = payload.session_id or str(current.get("session_id") or source_meta.get("session_id") or "")
    if not dataset_id:
        dataset_id = "legacy"
        if service.get_dataset(dataset_id) is None:
            service.create_dataset(dataset_id=dataset_id, name="Legacy takes", description="Auto-generated compatibility bucket")
    if not session_id:
        session_id = "legacy"
    if service.get_session(dataset_id, session_id) is None:
        service.create_session(dataset_id=dataset_id, session_id=session_id, name=session_id, description="Auto-generated compatibility session")
    updated = service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        updates=payload.model_dump(exclude_unset=True),
        source_metadata=source_meta,
    )
    return {"ok": True, "take_id": take_id, "metadata": updated}


@app.post("/api/takes/{take_id}/object-annotations")
def upsert_take_object_annotation(take_id: str, payload: ObjectAnnotationUpsertRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    service = DatasetService(settings.data_dir)
    source_meta = detail.metadata if isinstance(detail.metadata, dict) else {}
    take_meta = detail.take_metadata if isinstance(detail.take_metadata, dict) else {}
    dataset_id = str(take_meta.get("dataset_id") or "legacy")
    session_id = str(take_meta.get("session_id") or source_meta.get("session_id") or "legacy")
    if service.get_dataset(dataset_id) is None:
        service.create_dataset(dataset_id=dataset_id, name="Legacy takes", description="Auto-generated compatibility bucket")
    if service.get_session(dataset_id, session_id) is None:
        service.create_session(dataset_id=dataset_id, session_id=session_id, name=session_id, description="Auto-generated compatibility session")
    annotation = service.upsert_object_annotation(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        annotation=payload.model_dump(),
        source_metadata=source_meta,
    )
    return {"ok": True, "take_id": take_id, "annotation": annotation}


@app.post("/api/takes/{take_id}/process")
def process_take_for_pipeline(take_id: str, payload: ExecuteTakeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    take_dir = settings.incoming_dir / take_id
    if not take_dir.is_dir():
        raise HTTPException(status_code=404, detail="Take not found")
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")

    pipeline = next((item for item in list_pipelines() if item.get("id") == payload.pipeline_id), None)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {payload.pipeline_id}")

    if pipeline.get("execution_backend") == "process_service":
        template_id = PROCESS_SERVICE_PIPELINE_TO_TEMPLATE.get(payload.pipeline_id)
        if template_id is None:
            raise HTTPException(status_code=400, detail=f"No process template mapping configured for pipeline '{payload.pipeline_id}'.")
        source_image, source_filename = resolve_source_image_path(settings, detail)
        if source_image is None:
            raise HTTPException(status_code=400, detail="No compatible source image found for this take.")
        service = ProcessService(settings)
        instances = service.list_instances()
        instance = next(
            (
                item
                for item in instances
                if item.get("template_id") == template_id and item.get("supported_input_type") == "rgb_image"
            ),
            None,
        )
        if instance is None:
            instance = service.create_instance(
                name="RGB Steel Ball Classification (Auto)",
                template_id=template_id,
                supported_input_type="rgb_image",
            )
        executed = service.execute(instance["id"], image_path=source_image)
        run = executed["run"]
        return {
            "ok": True,
            "take_id": take_id,
            "pipeline_id": payload.pipeline_id,
            "pipeline_instance_id": executed["pipeline_instance"]["id"],
            "run_id": run["run_id"],
            "status": run["status"],
            "source_image": source_filename,
            "result_path": str((settings.data_dir / "processes" / "runs" / executed["pipeline_instance"]["id"] / run["run_id"]).resolve()),
        }

    output_dir = settings.processed_dir / take_id
    if payload.reprocess and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if payload.pipeline_id == "3d_ball_inspection":
            result = run_ball_inspection_flow(settings.data_dir, take_id=take_id)
            return {"ok": True, "take_id": take_id, "pipeline_id": payload.pipeline_id, "result_path": str(result.output_dir / "result.json")}
        config = load_processing_config(Path("config/processing.yaml"))
        result = process_take(take_dir, output_dir, config)
        (output_dir / "DONE").touch()
        return {"ok": True, "take_id": take_id, "pipeline_id": payload.pipeline_id, "result_path": str(output_dir / "result.json"), "status": result.status}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/takes/{take_id}/outputs")
def clear_take_outputs(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    output_dir = settings.processed_dir / take_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    return {"ok": True, "take_id": take_id}


@app.get("/api/takes/{take_id}/poc-summary")
def take_poc_summary(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None or detail.result is None:
        raise HTTPException(status_code=404, detail="Processed take not found")
    if isinstance(detail.result.get("poc_summary"), dict):
        return detail.result["poc_summary"]
    return build_poc_run_summary(
        detail.result,
        metadata=detail.metadata,
        output_dir=settings.processed_dir / take_id,
        engine=detail.result.get("processing_engine"),
    )


@app.get("/api/labels")
def labels(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"allowed_labels": sorted(ALLOWED_LABELS), "takes": list_labeled_takes(settings.data_dir)}


@app.get("/api/takes/{take_id}/labels")
def take_labels(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any] | None:
    if take_id not in {take.take_id for take in list_takes(settings)}:
        raise HTTPException(status_code=404, detail="Take not found")
    return load_labels(settings.data_dir, take_id)


@app.post("/api/takes/{take_id}/labels")
def update_take_labels(take_id: str, payload: LabelRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if take_id not in {take.take_id for take in list_takes(settings)}:
        raise HTTPException(status_code=404, detail="Take not found")
    try:
        return save_labels(settings.data_dir, take_id, payload.labels, notes=payload.notes, reviewer=payload.reviewer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/exports/labels")
def label_export(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return export_labeled_dataset_summary(settings.data_dir)


@app.get("/api/exports/object-metrics", response_model=None)
def object_metrics_export(
    take_id: str | None = None,
    format: str = "json",
    settings: ApiSettings = Depends(get_settings),
):
    rows = export_object_metrics(settings.data_dir, take_id=take_id)
    if format == "csv":
        return Response(_rows_to_csv(rows), media_type="text/csv")
    return rows


@app.get("/api/takes/{take_id}/validate-result")
def validate_take_result(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None or detail.result is None:
        raise HTTPException(status_code=404, detail="Processed take not found")
    validate_result_payload(detail.result)
    return {"take_id": take_id, "valid": True}


@app.get("/api/takes/{take_id}/files/{filename:path}")
def take_file(
    take_id: str,
    filename: str,
    settings: ApiSettings = Depends(get_settings),
) -> FileResponse:
    path = safe_take_file(settings, take_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/api/takes/{take_id}/source-histogram")
def take_source_histogram(
    take_id: str,
    filename: str | None = None,
    max_dim: int = 512,
    bins: int = 32,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    image_path, resolved_name = resolve_source_image_path(settings, detail, filename)
    if image_path is None or resolved_name is None:
        raise HTTPException(status_code=404, detail="No source image available for histogram.")
    try:
        return load_or_compute_histogram(settings, take_id, resolved_name, image_path, max_dim=max_dim, bins=bins)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Histogram unavailable for this source image: {exc}") from exc


@app.get("/api/events/stream")
async def events_stream(settings: ApiSettings = Depends(get_settings)) -> StreamingResponse:
    return StreamingResponse(
        stream_events(settings),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    handle = io.StringIO()
    fieldnames = sorted({key for row in rows for key in row.keys()})
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _runtime_config_payload() -> dict[str, Any]:
    status = read_runtime_config()
    return {
        "config": status.config.model_dump(mode="json"),
        "path": str(status.path),
        "default_calibration_file": status.default_calibration_file,
        "default_calibration_valid": status.default_calibration_valid,
        "warning": status.warning,
    }


def _preview_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }


def _age_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())
