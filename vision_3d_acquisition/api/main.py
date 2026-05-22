from __future__ import annotations

import csv
import io
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from vision_3d_acquisition.acquisition.preview import preview_image_path, read_preview_metadata
from vision_3d_acquisition.acquisition.usb_camera import (
    apply_camera_controls,
    camera_controls_snapshot,
    capture_image,
    discover_cameras,
    record_video,
)
from vision_3d_acquisition.acquisition.process_integration import trigger_bound_processing_for_take
from vision_3d_acquisition.acquisition.process_integration import (
    acquisition_processing_status as read_acquisition_processing_status,
    process_acquired_take_if_bound,
)
from vision_3d_acquisition.acquisition.processing_contracts import AcquiredTakeReady
from vision_3d_acquisition.acquisition.trispector_25d import TriSpectorFtpAcquisitionAdapter
from vision_3d_acquisition.acquisition.replay_dataset import ReplayableAcquisitionService
from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
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
from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing
from vision_3d_acquisition.api.schemas import DatasetSessionSummary, DatasetSummary, HealthResponse, RuntimeState, TakeDetail, TakeSummary
from vision_3d_acquisition.api.settings import ApiSettings, get_cors_origins, get_settings
from vision_3d_acquisition.poc.exports import export_labeled_dataset_summary, export_object_metrics
from vision_3d_acquisition.poc.labels import ALLOWED_LABELS, list_labeled_takes, load_labels, save_labels
from vision_3d_acquisition.poc.summary import build_poc_run_summary, validate_result_payload
from vision_3d_acquisition.pipelines.registry import list_pipelines
from vision_3d_acquisition.fusion.preparation import build_fusion_preview, resolve_fusion_inputs
from vision_3d_acquisition.fusion.published_service import PublishedInspectionResultService
from vision_3d_acquisition.fusion.service import FusionService
from vision_3d_acquisition.operations.summary import INDEX_PATH, read_json as read_operations_json
from vision_3d_acquisition.processes import ProcessService
from vision_3d_acquisition.processes.bindings import ProcessBindingService
from vision_3d_acquisition.runtime_config import (
    RuntimeConfig,
    clear_default_calibration,
    read_runtime_config,
    set_default_calibration,
    warn_if_deprecated_processing_local_calibration,
    write_runtime_config,
)
from vision_3d_acquisition.runtime.mjpeg import mjpeg_stream_frames
from vision_3d_acquisition.runtime.process_registry import ProcessHeartbeat, RuntimeProcessRegistry
from vision_3d_acquisition.runtime.supervisor import get_runtime_supervisor
from vision_3d_acquisition.runtime.worker_manager import get_runtime_worker_manager


app = FastAPI(title="3D Acquisition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_cors_origins()),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calibration_router)
_API_PROCESS_HEARTBEAT: ProcessHeartbeat | None = None


@app.on_event("startup")
def _warn_deprecated_processing_local_calibration() -> None:
    warn_if_deprecated_processing_local_calibration()


@app.on_event("startup")
def _start_api_process_heartbeat() -> None:
    global _API_PROCESS_HEARTBEAT
    settings = get_settings()
    hb = ProcessHeartbeat(
        RuntimeProcessRegistry(settings.data_dir),
        process_name="api_server",
        runtime_role="api",
        pid=os.getpid(),
        interval_seconds=3.0,
        base_metadata={"module": "vision_3d_acquisition.api.main"},
    )
    hb.start()
    _API_PROCESS_HEARTBEAT = hb


@app.on_event("shutdown")
def _stop_api_process_heartbeat() -> None:
    global _API_PROCESS_HEARTBEAT
    if _API_PROCESS_HEARTBEAT is not None:
        _API_PROCESS_HEARTBEAT.stop(status="stopped")
        _API_PROCESS_HEARTBEAT = None


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
    acquisition_group_id: str | None = None


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
    source_id: str | None = None
    modality: str | None = None
    purpose: str = "manual_debug"
    acquisition_group_id: str | None = None
    stage_params: dict[str, Any] | None = None


class CreateProcessBindingRequest(BaseModel):
    name: str
    source_id: str
    modality: str
    purpose: str
    pipeline_id: str
    active_recipe_version_id: str
    calibration_profile_id: str | None = None
    enabled: bool = True


class UpdateProcessBindingRequest(BaseModel):
    name: str | None = None
    source_id: str | None = None
    modality: str | None = None
    purpose: str | None = None
    pipeline_id: str | None = None
    active_recipe_version_id: str | None = None
    calibration_profile_id: str | None = None
    enabled: bool | None = None


class ActivateProcessBindingRecipeRequest(BaseModel):
    recipe_version_id: str


class PreviewSegmentationParams(BaseModel):
    threshold: int = 125
    auto_threshold: bool = False
    invert: bool = False
    morph_op: str = "open_close"
    erode_kernel_size: int = 3
    erode_iterations: int = 1
    dilate_kernel_size: int = 3
    dilate_iterations: int = 1
    close_kernel_size: int = 5
    fill_holes: bool = True
    min_area_px: int = 120
    max_area_px: int | None = None
    roi_enabled: bool = False


class PreviewSegmentationRequest(BaseModel):
    take_id: str
    pipeline_id: str
    stage_id: str = "segmentation"
    params: dict[str, Any] = Field(default_factory=dict)


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
    acquisition_group_id: str | None = None


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
    acquisition_group_id: str | None = None
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    preview_interval_ms: int = 250


class RegisterTriSpectorUploadRequest(BaseModel):
    upload_path: str
    source_id: str = "trispector_ftp_0"
    take_id: str | None = None


class StartReplaySessionRequest(BaseModel):
    dataset_id: str
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayTakeRequest(BaseModel):
    pipeline_id: str | None = None


class ReplaySessionRequest(BaseModel):
    dataset_id: str | None = None
    pipeline_id: str | None = None


class ArchiveTakeRequest(BaseModel):
    reason: str | None = None


class ResolveAcquisitionProcessingRequest(BaseModel):
    take_id: str
    source_id: str
    modality: str
    modality_family: str | None = None
    acquisition_process_id: str | None = None
    acquisition_run_id: str | None = None
    acquisition_group_id: str | None = None
    session_id: str | None = None
    asset_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    auto_process: bool = True
    purpose: str = "acquisition_inspection"


class PermanentDeleteTakeRequest(BaseModel):
    confirmation: str
    delete_runs: bool = True


class SourceControlsUpdateRequest(BaseModel):
    controls: dict[str, dict[str, Any]]


class CreateAcquisitionGroupRequest(BaseModel):
    name: str | None = None
    station_id: str | None = None
    trigger_id: str | None = None
    encoder_position: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateAcquisitionGroupRequest(BaseModel):
    name: str | None = None
    station_id: str | None = None
    trigger_id: str | None = None
    encoder_position: float | None = None
    completed_at: str | None = None
    status: str | None = None
    metadata: dict[str, Any] | None = None


class FuseAcquisitionGroupRequest(BaseModel):
    recipe_version_id: str | None = None
    force: bool = False


class CreateFusionRunRequest(BaseModel):
    force: bool = False
    auto_publish: bool = False
    recipe_version_id: str | None = None
    rules: dict[str, Any] = Field(default_factory=dict)


class PublishFusionResultRequest(BaseModel):
    fusion_run_id: str | None = None


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


@app.get("/api/process-bindings")
def list_process_bindings(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return ProcessBindingService(settings).list_bindings()


@app.post("/api/process-bindings")
def create_process_binding(payload: CreateProcessBindingRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ProcessBindingService(settings).create_binding(
            name=payload.name,
            source_id=payload.source_id,
            modality=payload.modality,
            purpose=payload.purpose,  # type: ignore[arg-type]
            pipeline_id=payload.pipeline_id,
            active_recipe_version_id=payload.active_recipe_version_id,
            calibration_profile_id=payload.calibration_profile_id,
            enabled=payload.enabled,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/process-bindings/{binding_id}")
def update_process_binding(binding_id: str, payload: UpdateProcessBindingRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    updates = {key: value for key, value in payload.model_dump().items() if value is not None}
    try:
        return ProcessBindingService(settings).update_binding(binding_id, updates)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/process-bindings/{binding_id}/activate-recipe")
def activate_process_binding_recipe(
    binding_id: str, payload: ActivateProcessBindingRecipeRequest, settings: ApiSettings = Depends(get_settings)
) -> dict[str, Any]:
    try:
        return ProcessBindingService(settings).activate_recipe(binding_id, payload.recipe_version_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/acquisition-groups")
def create_acquisition_group(payload: CreateAcquisitionGroupRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return AcquisitionGroupService(settings.data_dir).create_acquisition_group(
        name=payload.name,
        station_id=payload.station_id,
        trigger_id=payload.trigger_id,
        encoder_position=payload.encoder_position,
        metadata=payload.metadata,
    )


@app.get("/api/acquisition-groups/{group_id}")
def get_acquisition_group(group_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id)
    if not payload:
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    latest_fusion = FusionService(settings).latest_fusion_for_group(group_id)
    latest_published = PublishedInspectionResultService(settings.data_dir).latest_for_acquisition_group(group_id)
    payload = {
        **payload,
        "fusion": {
            "has_run": latest_fusion is not None,
            "latest_run_id": latest_fusion.get("fusion_run_id") if isinstance(latest_fusion, dict) else None,
            "latest_result": latest_fusion,
            "latest_published_result_id": latest_published.get("published_result_id") if isinstance(latest_published, dict) else None,
            "latest_published_status": latest_published.get("status") if isinstance(latest_published, dict) else None,
        },
    }
    return payload


@app.get("/api/acquisition-groups/{group_id}/takes")
def list_acquisition_group_takes(group_id: str, settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    try:
        return AcquisitionGroupService(settings.data_dir).list_group_takes(group_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/acquisition-groups/{group_id}/fusion-inputs")
def acquisition_group_fusion_inputs(group_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    return asdict(resolve_fusion_inputs(settings.data_dir, group_id))


@app.get("/api/acquisition-groups/{group_id}/fusion-preview")
def acquisition_group_fusion_preview(group_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    return asdict(build_fusion_preview(settings.data_dir, group_id))


@app.post("/api/acquisition-groups/{group_id}/fusion-runs")
def create_fusion_run(
    group_id: str,
    payload: CreateFusionRunRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    try:
        fusion_result = FusionService(settings).run_fusion(
            acquisition_group_id=group_id,
            force=payload.force,
            recipe_version_id=payload.recipe_version_id,
            rules=payload.rules,
        )
        response: dict[str, Any] = {
            "fusion_run_id": str(fusion_result.get("fusion_run_id") or ""),
            "result": fusion_result,
        }
        if payload.auto_publish:
            published = PublishedInspectionResultService(settings.data_dir).publish_from_fusion_result(fusion_result=fusion_result, persist=True)
            response["published_result_id"] = published["published_result_id"]
            response["published_result"] = published
        return response
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/acquisition-groups/{group_id}/fusion-runs")
def list_fusion_runs(group_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    runs = FusionService(settings).list_group_runs(group_id)
    return {"acquisition_group_id": group_id, "runs": runs}


@app.get("/api/fusion-runs/{fusion_run_id}")
def get_fusion_run(fusion_run_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = FusionService(settings).get_run(fusion_run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown fusion run: {fusion_run_id}")
    return payload


@app.get("/api/acquisition-groups/{group_id}/fusion-result/latest")
def latest_group_fusion_result(group_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    payload = FusionService(settings).latest_result_for_group(group_id)
    if payload is None:
        return {"acquisition_group_id": group_id, "status": "not_found", "run": None, "result": None}
    return payload


@app.post("/api/acquisition-groups/{group_id}/fuse")
def fuse_acquisition_group(
    group_id: str,
    payload: FuseAcquisitionGroupRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return create_fusion_run(
            group_id,
            CreateFusionRunRequest(force=payload.force, recipe_version_id=payload.recipe_version_id, rules={}),
            settings,
        )
    except HTTPException:
        raise


@app.post("/api/acquisition-groups/{group_id}/publish")
def publish_acquisition_group_result(
    group_id: str,
    payload: PublishFusionResultRequest | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    fusion_service = FusionService(settings)
    fusion_result: dict[str, Any] | None
    requested_run_id = payload.fusion_run_id if payload else None
    if requested_run_id:
        fusion_result = fusion_service.get_run(requested_run_id)
        if fusion_result is None:
            raise HTTPException(status_code=404, detail=f"Unknown fusion run: {requested_run_id}")
        if str(fusion_result.get("acquisition_group_id") or "") != group_id:
            raise HTTPException(status_code=400, detail="fusion_run_id does not belong to this acquisition group")
    else:
        fusion_result = fusion_service.latest_fusion_for_group(group_id)
    if fusion_result is None:
        raise HTTPException(status_code=404, detail="No fusion result available to publish")
    try:
        return PublishedInspectionResultService(settings.data_dir).publish_from_fusion_result(fusion_result=fusion_result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/acquisition-groups/{group_id}/fuse-and-publish")
def fuse_and_publish_acquisition_group(
    group_id: str,
    payload: FuseAcquisitionGroupRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    fusion_result = FusionService(settings).run_fusion(
        acquisition_group_id=group_id,
        force=payload.force,
        recipe_version_id=payload.recipe_version_id,
        rules={},
    )
    return PublishedInspectionResultService(settings.data_dir).publish_from_fusion_result(fusion_result=fusion_result)


@app.post("/api/takes/{take_id}/publish-25d-result")
def publish_25d_take_result(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    result = detail.result if isinstance(detail.result, dict) else None
    if not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="No processed result available for this take")
    pipeline = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
    if str(pipeline.get("pipeline_family") or "") != "25d":
        raise HTTPException(status_code=400, detail="Latest result for this take is not a 25D pipeline output")
    status = str(result.get("status") or "")
    if status not in {"ok", "success", "warning", "completed"}:
        raise HTTPException(status_code=400, detail=f"25D result status is not publishable: {status or 'unknown'}")
    return PublishedInspectionResultService(settings.data_dir).publish_from_25d_result(
        take_id=take_id,
        result_payload=result,
        persist=True,
    )


@app.get("/api/operator/inspection-results/latest")
def latest_operator_inspection_result(
    station_id: str | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    payload = PublishedInspectionResultService(settings.data_dir).get_latest_published_result(station_id=station_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No published inspection result found")
    return payload


@app.get("/api/operator/inspection-results/{published_result_id}")
def get_operator_inspection_result(published_result_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = PublishedInspectionResultService(settings.data_dir).get_published_result(published_result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown inspection result: {published_result_id}")
    return payload


@app.get("/api/operator/inspection-results")
def list_operator_inspection_results(
    station_id: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return {
        "results": PublishedInspectionResultService(settings.data_dir).list_published_results(
            station_id=station_id,
            session_id=session_id,
            limit=limit,
        ),
        "station_id": station_id,
        "session_id": session_id,
        "limit": limit,
    }


@app.get("/api/operations/cards")
def list_operations_cards(limit: int = 100, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    index = read_operations_json(settings.data_dir / INDEX_PATH) or {"cards": [], "updated_at": None, "latest_take_id": None}
    rows = [item for item in index.get("cards", []) if isinstance(item, dict)]
    max_items = max(1, min(int(limit), 500))
    return {
        "cards": rows[:max_items],
        "updated_at": index.get("updated_at"),
        "latest_take_id": index.get("latest_take_id"),
        "limit": max_items,
    }


@app.get("/api/inspection-results/latest")
def latest_published_inspection_result(station_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return latest_operator_inspection_result(station_id=station_id, settings=settings)


@app.get("/api/inspection-results/{published_result_id}")
def get_published_inspection_result(published_result_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return get_operator_inspection_result(published_result_id=published_result_id, settings=settings)


@app.get("/api/inspection-results")
def list_published_inspection_results(
    station_id: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return list_operator_inspection_results(
        station_id=station_id,
        session_id=session_id,
        limit=limit,
        settings=settings,
    )


@app.post("/api/acquisition-groups/{group_id}/takes/{take_id}")
def attach_take_acquisition_group(group_id: str, take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        metadata = AcquisitionGroupService(settings.data_dir).attach_take_to_group(take_id, group_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "take_id": take_id, "acquisition_group_id": group_id, "metadata": metadata}


@app.patch("/api/acquisition-groups/{group_id}")
def patch_acquisition_group(group_id: str, payload: UpdateAcquisitionGroupRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    updates = {key: value for key, value in payload.model_dump().items() if value is not None}
    try:
        return AcquisitionGroupService(settings.data_dir).update_acquisition_group(group_id, updates)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


@app.get("/api/sources/{source_id}/controls")
def source_controls(source_id: str) -> dict[str, Any]:
    if not source_id.startswith("usb_camera_"):
        controls = {}
        for name in ("exposure", "focus", "gain", "brightness", "contrast", "sharpness", "saturation", "white_balance"):
            controls[name] = {
                "supported": False,
                "readable": False,
                "writable": False,
                "value": None,
                "min": None,
                "max": None,
                "step": None,
                "default": None,
                "unit": None,
                "auto_supported": False,
                "auto_value": None,
                "backend_property": "",
                "range_source": "unknown",
                "warning": "Not supported by this camera/backend",
            }
        return {
            "source_id": source_id,
            "backend": None,
            "controls": controls,
            "warnings": ["controls_not_supported_for_source_type"],
            "diagnostics": {"raw_get": {}, "unsupported_properties": list(controls.keys()), "last_apply_errors": []},
        }
    try:
        index = int(source_id.rsplit("_", 1)[1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"code": "SOURCE_NOT_AVAILABLE", "message": f"Invalid source id: {source_id}"}) from exc
    try:
        snapshot = camera_controls_snapshot(index)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "SOURCE_NOT_AVAILABLE", "message": str(exc)}) from exc
    snapshot["source_id"] = source_id
    return snapshot


@app.post("/api/sources/{source_id}/controls")
def update_source_controls(source_id: str, payload: SourceControlsUpdateRequest) -> dict[str, Any]:
    if not source_id.startswith("usb_camera_"):
        raise HTTPException(status_code=400, detail={"code": "SOURCE_NOT_AVAILABLE", "message": "Source controls only available for usb_camera sources"})
    try:
        index = int(source_id.rsplit("_", 1)[1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"code": "SOURCE_NOT_AVAILABLE", "message": f"Invalid source id: {source_id}"}) from exc
    try:
        snapshot = apply_camera_controls(index, payload.controls)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "CAMERA_BUSY", "message": str(exc)}) from exc
    snapshot["source_id"] = source_id
    return snapshot


@app.post("/api/capture/image")
def capture_rgb_image(payload: UsbCaptureRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if payload.acquisition_group_id:
        if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(payload.acquisition_group_id):
            raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {payload.acquisition_group_id}")
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
            acquisition_group_id=payload.acquisition_group_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    source_id = f"usb_camera_{payload.camera_index}"
    processing = trigger_bound_processing_for_take(
        settings,
        take_id=take_id,
        source_id=source_id,
        modality="rgb",
        purpose="acquisition_inspection",
    )
    return {"take_id": take_id, "folder": str(folder), "modalities": ["rgb"], "acquisition_processing": processing}


@app.post("/api/takes/capture")
def capture_take_to_dataset(payload: CaptureTakeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = DatasetService(settings.data_dir)
    if service.get_dataset(payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if service.get_session(payload.dataset_id, payload.dataset_session_id) is None:
        raise HTTPException(status_code=404, detail="Dataset session not found")
    if payload.acquisition_group_id:
        if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(payload.acquisition_group_id):
            raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {payload.acquisition_group_id}")
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
            acquisition_group_id=payload.acquisition_group_id,
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
        source_metadata={"session_id": payload.dataset_session_id, "acquisition_group_id": payload.acquisition_group_id},
    )
    detail = get_take_detail(settings, take_id)
    source_id = f"usb_camera_{payload.camera_index}"
    processing = trigger_bound_processing_for_take(
        settings,
        take_id=take_id,
        source_id=source_id,
        modality="rgb",
        purpose="acquisition_inspection",
    )
    return {
        "ok": True,
        "take_id": take_id,
        "folder": str(folder),
        "modalities": ["rgb"],
        "acquisition_processing": processing,
        "metadata": sidecar,
        "take": detail.model_dump(mode="json") if detail else None,
    }


@app.post("/api/acquisition/trispector/register-upload")
def register_trispector_upload(payload: RegisterTriSpectorUploadRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    adapter = TriSpectorFtpAcquisitionAdapter(settings, source_id=payload.source_id)
    try:
        result = adapter.parse_and_register_trispector_upload(Path(payload.upload_path), take_id=payload.take_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **result}


@app.get("/api/acquisition/replay/active-session")
def active_replay_session(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"active_session": ReplayableAcquisitionService(settings.data_dir).active_session()}


@app.post("/api/acquisition/replay/sessions/start")
def start_replay_session(payload: StartReplaySessionRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if DatasetService(settings.data_dir).get_session(payload.dataset_id, payload.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset session: {payload.dataset_id}/{payload.session_id}")
    return ReplayableAcquisitionService(settings.data_dir).start_session(
        dataset_id=payload.dataset_id,
        session_id=payload.session_id,
        metadata=payload.metadata,
    )


@app.post("/api/acquisition/replay/sessions/stop")
def stop_replay_session(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"stopped_session": ReplayableAcquisitionService(settings.data_dir).stop_session()}


@app.get("/api/takes/{take_id}/replay-manifest")
def take_replay_manifest(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    path = settings.incoming_dir / take_id / "replay_manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Replay manifest not found for take: {take_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Replay manifest is not valid JSON") from exc
    return payload if isinstance(payload, dict) else {"manifest": payload}


@app.post("/api/replay/takes/{take_id}")
def replay_take_endpoint(take_id: str, payload: ReplayTakeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ReplayableAcquisitionService(settings.data_dir).replay_take(take_id=take_id, pipeline_id=payload.pipeline_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/replay/sessions/{session_id}")
def replay_session_endpoint(session_id: str, payload: ReplaySessionRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return ReplayableAcquisitionService(settings.data_dir).replay_session(
            session_id=session_id,
            dataset_id=payload.dataset_id,
            pipeline_id=payload.pipeline_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/acquisition/processing/resolve")
def resolve_acquisition_processing(payload: ResolveAcquisitionProcessingRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    decision = process_acquired_take_if_bound(
        settings,
        AcquiredTakeReady(
            take_id=payload.take_id,
            source_id=payload.source_id,
            modality=payload.modality,
            modality_family=payload.modality_family,
            acquisition_process_id=payload.acquisition_process_id,
            acquisition_run_id=payload.acquisition_run_id,
            acquisition_group_id=payload.acquisition_group_id,
            session_id=payload.session_id,
            asset_paths=dict(payload.asset_paths),
            metadata=dict(payload.metadata),
            warnings=list(payload.warnings),
        ),
        purpose=payload.purpose,
        auto_process=payload.auto_process,
    )
    body = {
        "ok": decision.ok,
        "status": decision.status,
        "normalized_modality": decision.normalized_modality,
        "original_modality": decision.original_modality,
        "pipeline_id": decision.pipeline_id,
        "recipe_version_id": decision.recipe_version_id,
        "calibration_profile_id": decision.calibration_profile_id,
        "warning": decision.warning,
        "warnings": decision.warnings,
        "run_ref": decision.run_ref,
        "metadata": decision.metadata,
    }
    return body


@app.get("/api/takes/{take_id}/acquisition-processing-status")
def take_acquisition_processing_status(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if not (settings.incoming_dir / take_id).is_dir() and not (settings.processed_dir / take_id).is_dir():
        raise HTTPException(status_code=404, detail="Take not found")
    return read_acquisition_processing_status(settings, take_id)


@app.post("/api/capture/video")
def capture_rgb_video(payload: UsbVideoCaptureRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if payload.acquisition_group_id:
        if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(payload.acquisition_group_id):
            raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {payload.acquisition_group_id}")
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
            acquisition_group_id=payload.acquisition_group_id,
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


@app.get("/api/runtime/stream/mjpeg")
def runtime_mjpeg_stream(source_id: str = "usb_camera_0", fps: float = 8.0, settings: ApiSettings = Depends(get_settings)) -> StreamingResponse:
    def _iter():
        yield from mjpeg_stream_frames(data_dir=settings.data_dir, source_id=source_id, fps=fps)

    return StreamingResponse(_iter(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/runtime/processes")
def runtime_processes(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    registry = RuntimeProcessRegistry(settings.data_dir, stale_after_seconds=15.0)
    return registry.list_processes()


@app.get("/api/runtime/processes/{process_id}")
def runtime_process_status(process_id: str) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.status(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runtime/processes/{process_id}/start")
def runtime_process_start(process_id: str) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.start(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runtime/processes/{process_id}/stop")
def runtime_process_stop(process_id: str) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.stop(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runtime/processes/{process_id}/restart")
def runtime_process_restart(process_id: str) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.restart(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runtime/processes/{process_id}/logs")
def runtime_process_logs(process_id: str, limit: int = 200) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.tail_logs(process_id, limit=max(1, min(limit, 2000)))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runtime/processes/{process_id}/events")
def runtime_process_events(process_id: str, limit: int = 200) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.events(process_id, limit=max(1, min(limit, 2000)))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runtime/processes/{process_id}/health-checklist")
def runtime_process_health_checklist(process_id: str) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.health_checklist(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runtime/processes/{process_id}/ftp-status")
def runtime_process_ftp_status(process_id: str) -> dict[str, Any]:
    supervisor = get_runtime_supervisor()
    try:
        return supervisor.ftp_status(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runtime/workers")
def runtime_workers() -> dict[str, Any]:
    manager = get_runtime_worker_manager()
    return {"workers": manager.list_worker_statuses(), "queue": manager.queue_status()}


@app.get("/api/runtime/workers/{worker_id}")
def runtime_worker_status(worker_id: str) -> dict[str, Any]:
    manager = get_runtime_worker_manager()
    payload = manager.get_worker_status(worker_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown runtime worker: {worker_id}")
    return payload


@app.get("/api/runtime/workers/{worker_id}/events")
def runtime_worker_events(worker_id: str, limit: int = 200) -> dict[str, Any]:
    manager = get_runtime_worker_manager()
    payload = manager.get_worker_status(worker_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown runtime worker: {worker_id}")
    return {"worker_id": worker_id, "events": manager.get_worker_events(worker_id, limit=max(1, min(limit, 5000)))}


@app.post("/api/runtime/workers/{worker_id}/stop-request")
def runtime_worker_stop_request(worker_id: str) -> dict[str, Any]:
    manager = get_runtime_worker_manager()
    payload = manager.get_worker_status(worker_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown runtime worker: {worker_id}")
    return manager.request_stop(worker_id)


@app.get("/api/runtime/queue")
def runtime_queue_status() -> dict[str, Any]:
    manager = get_runtime_worker_manager()
    return manager.queue_status()


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
    semantic_label: str | None = None,
    superclass_label: str | None = None,
    search: str | None = None,
    show_archived: bool = False,
) -> list[TakeSummary]:
    return list_takes(
        settings,
        session_id=session_id,
        dataset_id=dataset_id,
        validation_status=validation_status,
        tag=tag,
        semantic_label=semantic_label,
        superclass_label=superclass_label,
        search=search,
        include_archived=show_archived,
    )


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


@app.get("/api/datasets/{dataset_id}/label-summary")
def dataset_label_summary(dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = DatasetService(settings.data_dir)
    if service.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return service.label_summary(dataset_id)


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


@app.post("/api/takes/{take_id}/remove-from-dataset")
def remove_take_from_dataset(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
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
    updated = service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        updates={"dataset_id": None, "session_id": None},
        source_metadata=source_meta,
    )
    return {"ok": True, "take_id": take_id, "metadata": updated, "raw_deleted": False}


@app.post("/api/takes/{take_id}/archive")
def archive_take(take_id: str, payload: ArchiveTakeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
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
    updated = service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        updates={
            "archived": True,
            "archived_at": datetime.now(UTC).isoformat(),
            "archived_reason": payload.reason or None,
        },
        source_metadata=source_meta,
    )
    return {"ok": True, "take_id": take_id, "metadata": updated}


@app.post("/api/takes/{take_id}/restore")
def restore_take(take_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
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
    updated = service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        updates={"archived": False, "archived_at": None, "archived_reason": None},
        source_metadata=source_meta,
    )
    return {"ok": True, "take_id": take_id, "metadata": updated}


@app.post("/api/takes/{take_id}/permanent-delete")
def permanent_delete_take(take_id: str, payload: PermanentDeleteTakeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if payload.confirmation != take_id:
        raise HTTPException(status_code=400, detail="Confirmation must exactly match take id")
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    deleted: list[str] = []
    incoming_dir = settings.incoming_dir / take_id
    processed_dir = settings.processed_dir / take_id
    if incoming_dir.exists():
        shutil.rmtree(incoming_dir)
        deleted.append(str(incoming_dir))
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        deleted.append(str(processed_dir))
    service = DatasetService(settings.data_dir)
    dataset_id, session_id = service.resolve_take_membership(take_id)
    if dataset_id and session_id:
        take_meta_dir = settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "sessions" / f"session_{session_id}" / "takes" / take_id
        if take_meta_dir.exists():
            shutil.rmtree(take_meta_dir)
            deleted.append(str(take_meta_dir))
    if payload.delete_runs:
        index_path = settings.data_dir / "processes" / "index" / "runs.json"
        if index_path.is_file():
            try:
                payload_json = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                payload_json = {"entries": []}
            entries = payload_json.get("entries") if isinstance(payload_json.get("entries"), list) else []
            next_entries: list[dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("take_id") or "") == take_id:
                    path = entry.get("path")
                    if path:
                        run_dir = Path(str(path))
                        if run_dir.exists():
                            shutil.rmtree(run_dir, ignore_errors=True)
                            deleted.append(str(run_dir))
                    continue
                next_entries.append(entry)
            payload_json["entries"] = next_entries
            index_path.write_text(json.dumps(payload_json, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "take_id": take_id,
        "deleted_paths": deleted,
        "warning": "Permanent delete removes raw take files, processed outputs, and linked metadata.",
    }


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

    detail_meta = (detail.metadata or {}) if isinstance(detail.metadata, dict) else {}
    source_meta = detail_meta.get("source")
    resolved_source_id = None
    if isinstance(source_meta, dict):
        resolved_source_id = source_meta.get("source_id") or source_meta.get("id")
    elif isinstance(source_meta, str):
        resolved_source_id = source_meta
    source_id = payload.source_id or str(resolved_source_id or "")
    modality = payload.modality or (detail.modalities[0] if detail.modalities else None)
    binding = None
    binding_warning = None
    if source_id and modality:
        try:
            binding = ProcessBindingService(settings).resolve_process_binding(source_id, modality, payload.purpose)  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if binding is None:
        binding_warning = (
            f"No active ProcessBinding resolved for source_id={source_id or '<unknown>'}, "
            f"modality={modality or '<unknown>'}, purpose={payload.purpose}; falling back to request pipeline."
        )
    pipeline_id = str((binding or {}).get("pipeline_id") or payload.pipeline_id)
    recipe_version_id = (binding or {}).get("active_recipe_version_id")
    calibration_profile_id = (binding or {}).get("calibration_profile_id")
    detail_metadata = detail.metadata if isinstance(detail.metadata, dict) else {}
    take_management = detail.take_metadata if isinstance(detail.take_metadata, dict) else {}
    resolved_acquisition_group_id = (
        payload.acquisition_group_id
        or take_management.get("acquisition_group_id")
        or detail_metadata.get("acquisition_group_id")
    )

    try:
        response = dispatch_take_processing(
            settings=settings,
            take_id=take_id,
            pipeline_id=pipeline_id,
            reprocess=payload.reprocess,
            source_id=source_id or None,
            recipe_version_id=str(recipe_version_id) if recipe_version_id else None,
            acquisition_group_id=str(resolved_acquisition_group_id) if resolved_acquisition_group_id else None,
            calibration_profile_id=str(calibration_profile_id) if calibration_profile_id else None,
            stage_params=payload.stage_params or None,
        )
        if binding_warning:
            response["warning"] = binding_warning
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipelines/preview-segmentation")
def preview_segmentation(payload: PreviewSegmentationRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    detail = get_take_detail(settings, payload.take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    pipeline = next((item for item in list_pipelines() if item.get("id") == payload.pipeline_id), None)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {payload.pipeline_id}")
    if pipeline.get("execution_backend") != "process_service":
        raise HTTPException(status_code=400, detail="Segmentation preview is currently supported only for process_service pipelines.")

    template_id = PROCESS_SERVICE_PIPELINE_TO_TEMPLATE.get(payload.pipeline_id)
    if template_id is None:
        raise HTTPException(status_code=400, detail=f"No process template mapping configured for pipeline '{payload.pipeline_id}'.")
    source_image, source_filename = resolve_source_image_path(settings, detail)
    if source_image is None or source_filename is None:
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
    try:
        preview = service.preview_segmentation(instance["id"], image_path=source_image, stage_id=payload.stage_id, params=payload.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "take_id": payload.take_id,
        "pipeline_id": payload.pipeline_id,
        "pipeline_instance_id": instance["id"],
        "source_image": source_filename,
        **preview,
    }


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
