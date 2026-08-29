from __future__ import annotations

import csv
import io
import json
import logging
import os
import subprocess
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageOps
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
from vision_3d_acquisition.api.validation import router as validation_router
from vision_3d_acquisition.api.events import stream_events
from vision_3d_acquisition.api.filesystem import (
    count_take_dirs,
    get_session_summary,
    get_take_detail,
    get_take_summary,
    list_session_summaries,
    latest_take,
    list_takes,
    list_takes_paged,
    read_runtime_state,
    safe_take_file,
)
from vision_3d_acquisition.api.dataset_session_exports import DatasetSessionExportService
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.physical_objects import PhysicalObjectRegistry
from vision_3d_acquisition.api.histogram import load_or_compute_histogram, resolve_source_image_path
from vision_3d_acquisition.api.feature_analytics import (
    build_filter_options,
    build_distributions,
    build_feature_records,
    filter_records_by_feature_range,
    summarize_records_by_feature_range,
    summarize_feature_scope,
    summarize_feature_values,
    _feature_definitions,
)
from vision_3d_acquisition.api.pipeline_defaults_25d import (
    clear_25d_defaults,
    read_25d_defaults,
    write_25d_defaults,
)
from vision_3d_acquisition.api.processing_jobs import (
    ProcessingJobCancelResponse,
    ProcessingJobCreateRequest,
    get_processing_job_service,
)
from vision_3d_acquisition.api.feature_jobs import (
    FeatureJobRequest,
    get_feature_job_service,
)
from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing
from vision_3d_acquisition.api.schemas import DatasetSessionSummary, DatasetSummary, HealthResponse, RuntimeState, TakeDetail, TakeSummary, TakeSummaryPage
from vision_3d_acquisition.api.settings import ApiSettings, get_cors_origins, get_settings
from vision_3d_acquisition.poc.exports import export_labeled_dataset_summary, export_object_metrics
from vision_3d_acquisition.poc.labels import ALLOWED_LABELS, list_labeled_takes, load_labels, save_labels
from vision_3d_acquisition.poc.summary import build_poc_run_summary, validate_result_payload
from vision_3d_acquisition.pipelines.processing_units import (
    PROCESSING_UNIT_REGISTRY_VERSION,
    processing_unit_contract_fingerprint,
    validate_processing_unit_contracts,
)
from vision_3d_acquisition.pipelines.comparison import (
    compare_pipeline_sources,
    get_pipeline_comparison,
    list_pipeline_comparisons,
    resolve_comparison_file,
    resolve_pipeline_run_source,
)
from vision_3d_acquisition.pipelines.partial_rerun_plan import (
    list_partial_rerun_plans,
    persist_partial_rerun_plan,
    plan_partial_rerun,
)
from vision_3d_acquisition.pipelines.partial_rerun_execution import execute_partial_rerun
from vision_3d_acquisition.pipelines.run_lineage import (
    generate_run_comparison_to_parent,
    list_pipeline_runs,
    resolve_run_children,
    resolve_run_comparison_id,
    resolve_run_parent,
)
from vision_3d_acquisition.pipelines.recipes import RecipeService
from vision_3d_acquisition.pipelines.registry import list_pipelines, list_processing_unit_definitions
from vision_3d_acquisition.fusion.preparation import build_fusion_preview, resolve_fusion_inputs
from vision_3d_acquisition.fusion.published_service import PublishedInspectionResultService
from vision_3d_acquisition.fusion.service import FusionService
from vision_3d_acquisition.operations.summary import INDEX_PATH, read_json as read_operations_json
from vision_3d_acquisition.processes import ProcessService
from vision_3d_acquisition.processes.bindings import ProcessBindingService
from vision_3d_acquisition.ml import MLService
from vision_3d_acquisition.ml.ingestion import IngestionWizardService
from vision_3d_acquisition.ml.ml_set_summary import MLSetSummaryService
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
from vision_3d_acquisition.runtime.routing import RuntimeRoutingService


app = FastAPI(title="3D Acquisition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_cors_origins()),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calibration_router)
app.include_router(validation_router)
_API_PROCESS_HEARTBEAT: ProcessHeartbeat | None = None
logger = logging.getLogger(__name__)


def _feature_query_multi(request: Request, *keys: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for key in keys:
        for raw in request.query_params.getlist(key):
            for part in str(raw).split(","):
                item = part.strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                values.append(item)
    return values


def parse_feature_analytics_query(request: Request) -> dict[str, Any]:
    dataset_values = _feature_query_multi(request, "dataset_id", "dataset_ids", "dataset", "datasets")
    session_values = _feature_query_multi(request, "session_id", "session_ids", "session", "sessions")
    pipeline_values = _feature_query_multi(request, "pipeline_id", "pipeline_ids", "pipeline")
    calibration_values = _feature_query_multi(request, "calibration_id", "calibration_ids", "calibration")
    return {
        "dataset_id": dataset_values[0] if dataset_values else None,
        "ml_set_id": (request.query_params.get("ml_set_id") or "").strip() or None,
        "session_id": session_values[0] if session_values else None,
        "physical_object_ids": _feature_query_multi(request, "physical_object_ids", "physical_object_id"),
        "take_ids": _feature_query_multi(request, "take_ids", "take_id"),
        "raw_labels": _feature_query_multi(request, "raw_labels", "labels"),
        "normalized_classes": _feature_query_multi(request, "normalized_classes", "normalized_class"),
        "labeled_superclasses": _feature_query_multi(request, "labeled_superclasses"),
        "processed_superclasses": _feature_query_multi(request, "processed_superclasses", "superclasses", "superclass"),
        "processed_classes": _feature_query_multi(request, "processed_classes"),
        "validation_status": _feature_query_multi(request, "validation_status"),
        "split": _feature_query_multi(request, "split"),
        "pipeline_id": pipeline_values[0] if pipeline_values else None,
        "calibration_id": calibration_values[0] if calibration_values else None,
        "date_from": (request.query_params.get("date_from") or "").strip() or None,
        "date_to": (request.query_params.get("date_to") or "").strip() or None,
        "feature_selection": _feature_query_multi(request, "feature_selection"),
    }


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
    recipe_id: str | None = None
    recipe_version: int | None = None


class RecipeUpsertRequest(BaseModel):
    recipe_id: str | None = None
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    parameters_by_unit: dict[str, Any] | None = None
    stage_params: dict[str, Any] | None = None
    unit_enabled_state: dict[str, Any] | None = None
    artifact_policy: dict[str, Any] | None = None
    calibration_snapshot_reference: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    parent_recipe_id: str | None = None


class RecipeCloneRequest(BaseModel):
    name: str | None = None
    recipe_id: str | None = None


class RecipeValidateRequest(BaseModel):
    recipe_id: str | None = None
    parameters_by_unit: dict[str, Any] | None = None
    stage_params: dict[str, Any] | None = None


class ComparisonSourceRequest(BaseModel):
    type: str
    label: str | None = None
    take_id: str | None = None
    run_id: str | None = None
    recipe_id: str | None = None
    version: int | None = None
    stage_params: dict[str, Any] | None = None
    parameters_by_unit: dict[str, Any] | None = None


class ComparePipelineRequest(BaseModel):
    left: ComparisonSourceRequest
    right: ComparisonSourceRequest


class ParentRunResultRefRequest(BaseModel):
    take_id: str
    run_id: str = "latest"


class PartialRerunPlanRequest(BaseModel):
    selected_unit_id: str
    changed_parameters: dict[str, Any] | None = None
    current_recipe_snapshot: dict[str, Any] | None = None
    parent_run_result_ref: ParentRunResultRefRequest | None = None
    parent_run_result: dict[str, Any] | None = None


class PartialRerunExecuteOptionsRequest(BaseModel):
    compare_to_parent: bool = True


class PartialRerunExecuteRequest(BaseModel):
    plan_id: str | None = None
    plan: dict[str, Any] | None = None
    parent_run_ref: ParentRunResultRefRequest
    effective_recipe_snapshot: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = None
    options: PartialRerunExecuteOptionsRequest | None = None


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
    environment_metadata: dict[str, Any] = {}
    tags: list[str] = []
    notes: str | None = None
    session_type: str = "engineering"
    metadata: dict[str, Any] = {}


class TakeMetadataUpdateRequest(BaseModel):
    friendly_name: str | None = None
    labels: list[str] | None = None
    tags: list[str] | None = None
    notes: str | None = None
    expected_class: str | None = None
    expected_diameter_mm: float | None = None
    expected_count: int | None = None
    physical_object_id: str | None = None
    operator_notes: str | None = None
    session_notes: str | None = None
    categories: list[str] | None = None
    reference_type: str | None = None
    is_reference: bool | None = None
    is_golden_sample: bool | None = None
    validation_status: str | None = None
    dataset_id: str | None = None
    session_id: str | None = None
    acquisition_group_id: str | None = None


class TakeBulkFilters(BaseModel):
    dataset_id: str | None = None
    ml_set_id: str | None = None
    session_id: str | None = None
    validation_status: str | None = None
    search: str | None = None
    tag: str | None = None
    created_from: str | None = None
    created_to: str | None = None
    split: str | None = None
    expected_class: str | None = None
    calibration_linkage_only: bool | None = None
    physical_object_id: str | None = None
    raw_label: str | None = None
    normalized_class: str | None = None
    superclass: str | None = None
    membership_status: str | None = None


class TakeBulkMetadataRequest(BaseModel):
    mode: str
    action: str
    take_ids: list[str] = Field(default_factory=list)
    filters: TakeBulkFilters | None = None
    exclude_take_ids: list[str] = Field(default_factory=list)
    dataset_id: str | None = None
    session_id: str | None = None
    tags: list[str] | None = None
    validation_status: str | None = None
    split: str | None = None
    expected_class: str | None = None
    dry_run: bool = False


class DatasetMlSetRequest(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    notes: str | None = None
    split_strategy: str | None = None
    validation_requirements: list[str] = Field(default_factory=list)


class DatasetMlSetMembersRequest(BaseModel):
    mode: str
    action: str = "add"
    take_ids: list[str] = Field(default_factory=list)
    filters: TakeBulkFilters | None = None
    exclude_take_ids: list[str] = Field(default_factory=list)
    dry_run: bool = False


class DatasetMlSetMembershipUpdateRequest(BaseModel):
    take_ids: list[str] = Field(default_factory=list)
    split: str | None = None
    include: bool | None = None
    expected_class: str | None = None
    expected_subclass: str | None = None
    review_required: bool | None = None
    default_trainable: bool | None = None


class IngestionRunCreateRequest(BaseModel):
    dataset_id: str
    session_ids: list[str] = Field(default_factory=list)
    name: str | None = None


class IngestionTableRequest(BaseModel):
    content: str
    input_format: str = "csv"


class IngestionPolicyUpdateRequest(BaseModel):
    policy: dict[str, Any] = Field(default_factory=dict)


class IngestionMaterializeRequest(BaseModel):
    ml_set_id: str


class PhysicalObjectReconcileRequest(BaseModel):
    dataset_id: str
    rows: list[dict[str, Any]] = Field(default_factory=list)


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
    persist_on_restart: bool = False
    replay_mode: str = "routing_override"
    auto_start_on_startup: bool = False
    replay_source: str = "offline_dataset_replay"


class ReplayTakeRequest(BaseModel):
    pipeline_id: str | None = None


class ReplaySessionRequest(BaseModel):
    dataset_id: str | None = None
    pipeline_id: str | None = None


class UpdateReplayConfigRequest(BaseModel):
    dataset_id: str | None = None
    session_id: str | None = None
    persist_on_restart: bool | None = None
    auto_start_on_startup: bool | None = None
    replay_mode: str | None = None
    replay_source: str | None = None
    metadata: dict[str, Any] | None = None


class MLDatasetRequest(BaseModel):
    id: str
    name: str
    description: str | None = None
    source_dataset_ids: list[str] = []
    filters: dict[str, Any] = {}
    label_schema: dict[str, Any] = {}
    object_count: int = 0
    tags: list[str] = []
    notes: str | None = None


class MLExperimentRequest(BaseModel):
    id: str
    name: str
    classifier_type: str
    classifier_types: list[str] = []
    experiment_mode: str = "single"
    feature_set_id: str
    dataset_id: str
    split_strategy: str = "stratified"
    training_config: dict[str, Any] = {}


class MLEvaluateRequest(BaseModel):
    id: str
    name: str
    classifier_type: str
    classifier_types: list[str] = []
    experiment_mode: str = "single"
    feature_set_id: str
    dataset_id: str
    split_strategy: str = "stratified"
    training_config: dict[str, Any] = {}


class PromoteModelRequest(BaseModel):
    notes: str | None = None


class MLDeploymentRequest(BaseModel):
    id: str
    model_id: str
    target_pipeline_id: str
    target_stage_id: str
    pipeline_family: str = "25d"
    deployed_by: str = "system"
    notes: str | None = None


class MLSmokeRunRequest(BaseModel):
    force_invalid_model: bool = False
    ci_mode: bool = True


class RuntimeRoutingRequest(BaseModel):
    dataset_id: str
    session_id: str
    routing_mode: str = "explicit_acquisition_destination"
    source: str = "runtime_ui"
    updated_by: str = "runtime_ui"


class RuntimeDefaultDestinationRequest(BaseModel):
    default_dataset_id: str
    default_session_id: str
    auto_create_session: bool = False
    session_naming_policy: str = "explicit_session"
    updated_by: str = "runtime_ui"


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


@app.get("/api/pipelines/{pipeline_id}/processing-units")
def pipeline_processing_units(pipeline_id: str) -> dict[str, Any]:
    units = list_processing_unit_definitions(pipeline_id)
    return {
        "pipeline_id": pipeline_id,
        "processing_units": units,
        "processing_unit_contract_version": PROCESSING_UNIT_REGISTRY_VERSION,
        "processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units),
        "validation": validate_processing_unit_contracts(units),
    }


@app.get("/api/pipelines/{pipeline_id}/recipes")
def pipeline_recipes(pipeline_id: str, include_archived: bool = False, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "recipes": RecipeService(settings).list_recipes(pipeline_id, include_archived=include_archived),
    }


@app.post("/api/pipelines/{pipeline_id}/partial-rerun/plan")
def pipeline_partial_rerun_plan(
    pipeline_id: str,
    payload: PartialRerunPlanRequest,
    persist: bool = False,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    # Keep direct Python callers compatible with the older positional calling style:
    # pipeline_partial_rerun_plan(pipeline_id, payload, settings)
    if isinstance(persist, ApiSettings) and not isinstance(settings, ApiSettings):
        settings = persist
        persist = False
    units = list_processing_unit_definitions(pipeline_id)
    parent_run_result: dict[str, Any] | None = None
    resolver_error: str | None = None
    if payload.parent_run_result_ref is not None:
        resolved = resolve_pipeline_run_source(
            settings,
            pipeline_id=pipeline_id,
            take_id=payload.parent_run_result_ref.take_id,
            run_id=payload.parent_run_result_ref.run_id,
        )
        if resolved is None:
            resolver_error = (
                f"Parent run result could not be resolved for take {payload.parent_run_result_ref.take_id} "
                f"run {payload.parent_run_result_ref.run_id}."
            )
        else:
            parent_run_result = dict(resolved.get("result_payload") or {})
            parent_run_result["artifacts"] = list(resolved.get("artifacts") or [])
            if isinstance(resolved.get("processing_unit_trace"), dict):
                parent_run_result["processing_unit_trace"] = dict(resolved.get("processing_unit_trace") or {})
    elif isinstance(payload.parent_run_result, dict):
        parent_run_result = dict(payload.parent_run_result)
    plan = plan_partial_rerun(
        pipeline_id,
        payload.selected_unit_id,
        payload.changed_parameters,
        payload.current_recipe_snapshot,
        parent_run_result,
        units,
    )
    if resolver_error:
        plan["safe"] = False
        plan["mode"] = "full_run"
        plan["fallback_mode"] = "full_run"
        plan.setdefault("blocking_reasons", []).append(resolver_error)
        plan.setdefault("explanation", {})
        plan["explanation"]["summary"] = resolver_error
        plan["explanation"]["fallback_reason"] = "Full-run fallback was selected because the parent run reference could not be resolved."
        plan["plan_quality"] = "unsafe"
        plan["confidence"] = "low"
    response = {
        "pipeline_id": pipeline_id,
        **plan,
    }
    if persist:
        persisted = persist_partial_rerun_plan(settings.data_dir, pipeline_id, response)
        response.update({
            "plan_id": persisted.get("plan_id"),
            "created_at": persisted.get("created_at"),
            "plan_path": persisted.get("plan_path"),
        })
    return response


@app.get("/api/pipelines/{pipeline_id}/partial-rerun/plans")
def pipeline_partial_rerun_plans(pipeline_id: str, limit: int = 10, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "plans": list_partial_rerun_plans(settings.data_dir, pipeline_id, limit=limit),
    }


@app.post("/api/pipelines/{pipeline_id}/partial-rerun/execute")
def pipeline_partial_rerun_execute(
    pipeline_id: str,
    payload: PartialRerunExecuteRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return execute_partial_rerun(
        settings,
        pipeline_id=pipeline_id,
        plan_id=payload.plan_id,
        plan_payload=payload.plan,
        parent_run_ref=payload.parent_run_ref.model_dump(mode="json") if payload.parent_run_ref else None,
        effective_recipe_snapshot=payload.effective_recipe_snapshot,
        overrides=payload.overrides,
        compare_to_parent=bool((payload.options or PartialRerunExecuteOptionsRequest()).compare_to_parent),
    )


@app.post("/api/pipelines/{pipeline_id}/recipes")
def create_pipeline_recipe(pipeline_id: str, payload: RecipeUpsertRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return RecipeService(settings).create_recipe(
            pipeline_id,
            name=payload.name,
            description=payload.description,
            recipe_id=payload.recipe_id,
            tags=payload.tags,
            notes=payload.notes,
            parameters_by_unit=payload.parameters_by_unit,
            stage_params=payload.stage_params,
            unit_enabled_state=payload.unit_enabled_state,
            artifact_policy=payload.artifact_policy,
            calibration_snapshot_reference=payload.calibration_snapshot_reference,
            provenance=payload.provenance,
            parent_recipe_id=payload.parent_recipe_id,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipelines/{pipeline_id}/recipes/from-current")
def create_pipeline_recipe_from_current(
    pipeline_id: str,
    payload: RecipeUpsertRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        provenance = {
            **payload.provenance,
            "source": "from_current",
        }
        return RecipeService(settings).create_recipe(
            pipeline_id,
            name=payload.name,
            description=payload.description,
            recipe_id=payload.recipe_id,
            tags=payload.tags,
            notes=payload.notes,
            parameters_by_unit=payload.parameters_by_unit,
            stage_params=payload.stage_params,
            unit_enabled_state=payload.unit_enabled_state,
            artifact_policy=payload.artifact_policy,
            calibration_snapshot_reference=payload.calibration_snapshot_reference,
            provenance=provenance,
            parent_recipe_id=payload.parent_recipe_id,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipelines/{pipeline_id}/recipes/{recipe_id}")
def get_pipeline_recipe(
    pipeline_id: str,
    recipe_id: str,
    version: int | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return RecipeService(settings).get_recipe(pipeline_id, recipe_id, version=version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/pipelines/{pipeline_id}/recipes/{recipe_id}")
def update_pipeline_recipe(
    pipeline_id: str,
    recipe_id: str,
    payload: RecipeUpsertRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return RecipeService(settings).update_recipe(
            pipeline_id,
            recipe_id,
            name=payload.name,
            description=payload.description,
            tags=payload.tags,
            notes=payload.notes,
            parameters_by_unit=payload.parameters_by_unit,
            stage_params=payload.stage_params,
            unit_enabled_state=payload.unit_enabled_state,
            artifact_policy=payload.artifact_policy,
            calibration_snapshot_reference=payload.calibration_snapshot_reference,
            provenance=payload.provenance,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipelines/{pipeline_id}/recipes/{recipe_id}/clone")
def clone_pipeline_recipe(
    pipeline_id: str,
    recipe_id: str,
    payload: RecipeCloneRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return RecipeService(settings).clone_recipe(
            pipeline_id,
            recipe_id,
            new_name=payload.name,
            new_recipe_id=payload.recipe_id,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipelines/{pipeline_id}/recipes/{recipe_id}/archive")
def archive_pipeline_recipe(pipeline_id: str, recipe_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return RecipeService(settings).archive_recipe(pipeline_id, recipe_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pipelines/{pipeline_id}/recipes/validate")
def validate_pipeline_recipe(
    pipeline_id: str,
    payload: RecipeValidateRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return RecipeService(settings).validate_recipe(
        pipeline_id,
        recipe_id=payload.recipe_id,
        parameters_by_unit=payload.parameters_by_unit,
        stage_params=payload.stage_params,
    )


@app.post("/api/pipelines/{pipeline_id}/compare")
def compare_pipeline_contracts(
    pipeline_id: str,
    payload: ComparePipelineRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return compare_pipeline_sources(
            settings,
            pipeline_id=pipeline_id,
            left=payload.left.model_dump(exclude_none=True),
            right=payload.right.model_dump(exclude_none=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipelines/{pipeline_id}/comparisons")
def pipeline_comparisons(
    pipeline_id: str,
    take_id: str | None = None,
    limit: int = 20,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "comparisons": list_pipeline_comparisons(settings, pipeline_id, take_id=take_id, limit=limit),
    }


@app.get("/api/pipelines/{pipeline_id}/comparisons/{comparison_id}")
def pipeline_comparison(
    pipeline_id: str,
    comparison_id: str,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return get_pipeline_comparison(settings, pipeline_id, comparison_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pipelines/{pipeline_id}/runs")
def pipeline_runs(
    pipeline_id: str,
    take_id: str | None = None,
    limit: int = 50,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "runs": list_pipeline_runs(settings, pipeline_id, take_id=take_id, limit=limit),
    }


@app.get("/api/pipelines/{pipeline_id}/runs/{run_id}/lineage")
def pipeline_run_lineage(
    pipeline_id: str,
    run_id: str,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "run_id": run_id,
        "parent": resolve_run_parent(settings, pipeline_id, run_id),
        "children": resolve_run_children(settings, pipeline_id, run_id),
        "comparison_id": resolve_run_comparison_id(settings, pipeline_id, run_id),
    }


@app.post("/api/pipelines/{pipeline_id}/runs/{run_id}/generate-comparison")
def pipeline_run_generate_comparison(
    pipeline_id: str,
    run_id: str,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return generate_run_comparison_to_parent(settings, pipeline_id, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def _acquisition_group_ready_for_fusion(settings: ApiSettings, group_id: str) -> bool:
    """Mirrors FusionService.run_fusion's own candidate-gathering so the
    readiness gate agrees with what fusion will actually find."""
    has_rgb = False
    has_25d = False
    for item in AcquisitionGroupService(settings.data_dir).list_group_takes(group_id):
        take_id = str(item.get("take_id") or "")
        if not take_id:
            continue
        detail = get_take_detail(settings, take_id)
        if detail is None or not isinstance(detail.result, dict):
            continue
        result = detail.result
        family = str(((result.get("processing_pipeline") or {}).get("pipeline_family") or ""))
        status = str(result.get("status") or "")
        if status not in {"ok", "success", "warning", "completed"}:
            continue
        candidates = [c for c in (result.get("object_candidates") or []) if isinstance(c, dict)]
        if not candidates:
            continue
        if family == "2d":
            has_rgb = True
        elif family == "25d":
            has_25d = True
    return has_rgb and has_25d


@app.post("/api/acquisition-groups/{group_id}/fusion-runs")
def create_fusion_run(
    group_id: str,
    payload: CreateFusionRunRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    if not payload.force and not _acquisition_group_ready_for_fusion(settings, group_id):
        raise HTTPException(
            status_code=409,
            detail="Acquisition group is not ready for fusion: missing successful 2D or 2.5D object candidates. Pass force=true to fuse anyway.",
        )
    try:
        fusion_service = FusionService(settings)
        fusion_result = fusion_service.run_fusion(
            acquisition_group_id=group_id,
            force=payload.force,
            recipe_version_id=payload.recipe_version_id,
            rules=payload.rules,
        )
        run_dir = fusion_service.runs_dir / str(fusion_result.get("fusion_run_id") or "")
        response: dict[str, Any] = {
            "ok": True,
            "fusion_run_id": str(fusion_result.get("fusion_run_id") or ""),
            "run": {"result_path": str(run_dir / "fusion_result.json")},
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
    if not AcquisitionGroupService(settings.data_dir).get_acquisition_group(group_id):
        raise HTTPException(status_code=404, detail=f"Unknown acquisition group: {group_id}")
    try:
        return FusionService(settings).run_fusion(
            acquisition_group_id=group_id,
            force=payload.force,
            recipe_version_id=payload.recipe_version_id,
            rules={},
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    return {"active_session": ReplayableAcquisitionService(settings.data_dir).session_state()}


@app.post("/api/acquisition/replay/sessions/start")
def start_replay_session(payload: StartReplaySessionRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if DatasetService(settings.data_dir).get_session(payload.dataset_id, payload.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset session: {payload.dataset_id}/{payload.session_id}")
    return ReplayableAcquisitionService(settings.data_dir).start_session(
        dataset_id=payload.dataset_id,
        session_id=payload.session_id,
        metadata=payload.metadata,
        persist_on_restart=payload.persist_on_restart,
        replay_mode=payload.replay_mode,
        auto_start_on_startup=payload.auto_start_on_startup,
        replay_source=payload.replay_source,
    )


@app.post("/api/acquisition/replay/sessions/stop")
def stop_replay_session(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"stopped_session": ReplayableAcquisitionService(settings.data_dir).stop_session()}


@app.post("/api/runtime/replay/start")
def runtime_replay_start(payload: StartReplaySessionRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return start_replay_session(payload, settings=settings)


@app.post("/api/runtime/replay/stop")
def runtime_replay_stop(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return stop_replay_session(settings=settings)


@app.post("/api/runtime/replay/pause")
def runtime_replay_pause(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"paused_session": ReplayableAcquisitionService(settings.data_dir).pause_session()}


@app.post("/api/runtime/replay/clear")
def runtime_replay_clear(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"cleared": ReplayableAcquisitionService(settings.data_dir).clear_session()}


@app.put("/api/runtime/replay/config")
def runtime_replay_config(payload: UpdateReplayConfigRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = ReplayableAcquisitionService(settings.data_dir)
    try:
        updated = service.update_session_config(
            dataset_id=payload.dataset_id,
            session_id=payload.session_id,
            persist_on_restart=payload.persist_on_restart,
            auto_start_on_startup=payload.auto_start_on_startup,
            replay_mode=payload.replay_mode,
            replay_source=payload.replay_source,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": updated}


@app.get("/api/runtime/routing")
def runtime_routing_state(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    routing = RuntimeRoutingService(settings.data_dir)
    replay_state = ReplayableAcquisitionService(settings.data_dir).session_state()
    resolved = routing.resolve_routing(replay_session=replay_state)
    return {
        "live_routing": routing.get_live_state(),
        "persistent_default": routing.get_default_destination(),
        "resolved": resolved,
        "replay_override_active": bool(replay_state and replay_state.get("active")),
    }


@app.put("/api/runtime/routing")
def runtime_routing_update(payload: RuntimeRoutingRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if DatasetService(settings.data_dir).get_session(payload.dataset_id, payload.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset session: {payload.dataset_id}/{payload.session_id}")
    routing = RuntimeRoutingService(settings.data_dir)
    updated = routing.set_live_state(
        dataset_id=payload.dataset_id,
        session_id=payload.session_id,
        mode=payload.routing_mode,
        source=payload.source,
        updated_by=payload.updated_by,
    )
    return {"live_routing": updated}


@app.post("/api/runtime/routing/activate")
def runtime_routing_activate(payload: RuntimeRoutingRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return runtime_routing_update(payload, settings=settings)


@app.post("/api/runtime/routing/clear")
def runtime_routing_clear(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    routing = RuntimeRoutingService(settings.data_dir)
    return {"cleared": routing.clear_live_state()}


@app.get("/api/runtime/default-destination")
def runtime_default_destination(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"default_destination": RuntimeRoutingService(settings.data_dir).get_default_destination()}


@app.put("/api/runtime/default-destination")
def runtime_default_destination_update(payload: RuntimeDefaultDestinationRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    if DatasetService(settings.data_dir).get_session(payload.default_dataset_id, payload.default_session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset session: {payload.default_dataset_id}/{payload.default_session_id}")
    routing = RuntimeRoutingService(settings.data_dir)
    updated = routing.set_default_destination(
        default_dataset_id=payload.default_dataset_id,
        default_session_id=payload.default_session_id,
        auto_create_session=payload.auto_create_session,
        session_naming_policy=payload.session_naming_policy,
        updated_by=payload.updated_by,
    )
    return {"default_destination": updated}


@app.delete("/api/runtime/default-destination")
def runtime_default_destination_clear(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"cleared": RuntimeRoutingService(settings.data_dir).clear_default_destination()}


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


@app.get("/api/pipeline-defaults/25d")
def get_pipeline_defaults_25d(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"defaults": read_25d_defaults(settings.state_dir)}


@app.put("/api/pipeline-defaults/25d")
def put_pipeline_defaults_25d(payload: dict[str, Any], settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    saved = write_25d_defaults(settings.state_dir, payload or {})
    return {"defaults": saved}


@app.delete("/api/pipeline-defaults/25d")
def delete_pipeline_defaults_25d(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    clear_25d_defaults(settings.state_dir)
    return {"defaults": {}}


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
    physical_object_id: str | None = None,
    session_type: str | None = None,
    category: str | None = None,
    reference_type: str | None = None,
    is_reference: bool | None = None,
    is_golden_sample: bool | None = None,
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
        physical_object_id=physical_object_id,
        session_type=session_type,
        category=category,
        reference_type=reference_type,
        is_reference=is_reference,
        is_golden_sample=is_golden_sample,
        include_archived=show_archived,
    )


@app.get("/api/takes/paged", response_model=TakeSummaryPage)
def takes_paged(
    settings: ApiSettings = Depends(get_settings),
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
    dataset_id: str | None = None,
    validation_status: str | None = None,
    tag: str | None = None,
    semantic_label: str | None = None,
    superclass_label: str | None = None,
    search: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    expected_class: str | None = None,
    split: str | None = None,
    calibration_linkage_only: bool = False,
    physical_object_id: str | None = None,
    session_type: str | None = None,
    category: str | None = None,
    reference_type: str | None = None,
    is_reference: bool | None = None,
    is_golden_sample: bool | None = None,
    show_archived: bool = False,
    profile: bool = False,
) -> TakeSummaryPage:
    payload = list_takes_paged(
        settings,
        limit=limit,
        offset=offset,
        session_id=session_id,
        dataset_id=dataset_id,
        validation_status=validation_status,
        tag=tag,
        semantic_label=semantic_label,
        superclass_label=superclass_label,
        search=search,
        created_from=created_from,
        created_to=created_to,
        expected_class=expected_class,
        split=split,
        calibration_linkage_only=calibration_linkage_only,
        physical_object_id=physical_object_id,
        session_type=session_type,
        category=category,
        reference_type=reference_type,
        is_reference=is_reference,
        is_golden_sample=is_golden_sample,
        include_archived=show_archived,
        profile=profile,
    )
    return TakeSummaryPage(**payload)


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
    take_counts = service.visible_take_counts_by_session(dataset_id, include_archived=False)
    payload: list[DatasetSessionSummary] = []
    for item in sessions:
        sid = str(item.get("id") or "")
        take_count = int(take_counts.get(sid) or 0)
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
        session_type=payload.session_type,
        metadata=payload.metadata,
        tags=payload.tags,
        notes=payload.notes,
        session_id=payload.id,
    )
    return DatasetSessionSummary(**created, take_count=0)


@app.get("/api/datasets/{dataset_id}/ml-sets")
def dataset_ml_sets(dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    service = DatasetService(settings.data_dir)
    if service.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    rows: list[dict[str, Any]] = []
    for item in service.list_ml_sets(dataset_id=dataset_id):
        ml_set_id = str(item.get("id") or "")
        memberships = service.list_ml_set_memberships(dataset_id=dataset_id, ml_set_id=ml_set_id) if ml_set_id else []
        rows.append({**item, **service.summarize_ml_set_memberships(item, memberships)})
    return rows


@app.get("/api/physical-objects")
def list_physical_objects(
    dataset_id: str,
    normalized_class: str | None = None,
    superclass: str | None = None,
    session_id: str | None = None,
    needs_review: bool | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> list[dict[str, Any]]:
    if DatasetService(settings.data_dir).get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return PhysicalObjectRegistry(settings).list_objects(
        dataset_id,
        normalized_class=normalized_class,
        superclass=superclass,
        session_id=session_id,
        needs_review=needs_review,
    )


@app.get("/api/physical-objects/{physical_object_id}")
def get_physical_object(physical_object_id: str, dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = PhysicalObjectRegistry(settings).get_object(dataset_id, physical_object_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Physical object not found")
    return payload


@app.get("/api/physical-objects/{physical_object_id}/takes")
def get_physical_object_takes(physical_object_id: str, dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return {"takes": PhysicalObjectRegistry(settings).object_takes(dataset_id, physical_object_id)}


@app.get("/api/physical-objects/{physical_object_id}/repeatability")
def get_physical_object_repeatability(physical_object_id: str, dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return PhysicalObjectRegistry(settings).object_repeatability(dataset_id, physical_object_id)


@app.post("/api/physical-objects/reconcile")
def reconcile_physical_objects(payload: PhysicalObjectReconcileRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    registry = PhysicalObjectRegistry(settings)
    synced = registry.sync_from_manifest_rows(payload.dataset_id, payload.rows)
    return {"dataset_id": payload.dataset_id, "synced_count": len(synced), "objects": synced}


@app.post("/api/datasets/{dataset_id}/ml-sets")
def create_dataset_ml_set(dataset_id: str, payload: DatasetMlSetRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = DatasetService(settings.data_dir)
    dataset = service.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    did = payload.id.strip() if payload.id else f"ml_set_{ts}"
    created = service.create_ml_set(
        dataset_id=dataset_id,
        ml_set_id=did,
        name=payload.name,
        task_type="classification",
        description=payload.description,
        notes=payload.notes,
        overwrite=False,
    )
    ml_set_path = settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "ml_sets" / f"ml_set_{did}" / "ml_set.json"
    existing = read_operations_json(ml_set_path) or {}
    existing = existing if isinstance(existing, dict) else {}
    existing["membership"] = existing.get("membership") or {"mode": "ids", "take_ids": [], "filter_snapshot": {}, "exclude_take_ids": []}
    existing["semantics"] = {
        "source": "manual_bulk_selection",
        "validation_requirements": payload.validation_requirements or [],
        "split_strategy": payload.split_strategy,
        "notes": payload.notes or "",
    }
    existing["updated_at"] = datetime.now(UTC).isoformat()
    ml_set_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    memberships = service.list_ml_set_memberships(dataset_id=dataset_id, ml_set_id=did)
    return {**created, **existing, **service.summarize_ml_set_memberships(existing, memberships)}


@app.post("/api/datasets/{dataset_id}/ml-sets/{ml_set_id}/members")
def add_dataset_ml_set_members(dataset_id: str, ml_set_id: str, payload: DatasetMlSetMembersRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = DatasetService(settings.data_dir)
    if service.get_ml_set(dataset_id, ml_set_id) is None:
        raise HTTPException(status_code=404, detail="ML set not found")
    candidate_ids = _resolve_bulk_take_ids(payload.mode, payload.take_ids, payload.filters, settings)
    excluded = {str(item).strip() for item in payload.exclude_take_ids if str(item).strip()}
    target_ids = [take_id for take_id in candidate_ids if take_id and take_id not in excluded]
    if payload.dry_run:
        return {
            "ok": True,
            "matched_count": len(candidate_ids),
            "affected_count": len(target_ids),
            "skipped_count": max(0, len(candidate_ids) - len(target_ids)),
            "failed_count": 0,
            "failed_ids": [],
        }
    affected_count = 0
    skipped_count = 0
    failed_ids: list[str] = []
    for take_id in target_ids:
        try:
            if payload.action == "remove":
                removed = service.remove_take_from_ml_set(dataset_id=dataset_id, ml_set_id=ml_set_id, take_id=take_id)
                if removed:
                    affected_count += 1
                else:
                    skipped_count += 1
            else:
                service.add_take_to_ml_set(dataset_id=dataset_id, ml_set_id=ml_set_id, take_id=take_id)
                affected_count += 1
        except Exception:
            failed_ids.append(take_id)
    ml_set_path = settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "ml_sets" / f"ml_set_{ml_set_id}" / "ml_set.json"
    existing = read_operations_json(ml_set_path) or {}
    existing = existing if isinstance(existing, dict) else {}
    existing["membership"] = {
        "mode": "ids" if payload.mode == "ids" else "filter_snapshot",
        "take_ids": target_ids if payload.mode == "ids" else [],
        "filter_snapshot": payload.filters.model_dump(exclude_none=True) if payload.filters else {},
        "exclude_take_ids": list(excluded),
    }
    existing["updated_at"] = datetime.now(UTC).isoformat()
    ml_set_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "matched_count": len(candidate_ids),
        "affected_count": affected_count,
        "skipped_count": skipped_count + max(0, len(candidate_ids) - len(target_ids)),
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids[:100],
    }


@app.post("/api/datasets/{dataset_id}/ml-sets/{ml_set_id}/members/update")
def update_dataset_ml_set_memberships(
    dataset_id: str,
    ml_set_id: str,
    payload: DatasetMlSetMembershipUpdateRequest,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    service = DatasetService(settings.data_dir)
    if service.get_ml_set(dataset_id, ml_set_id) is None:
        raise HTTPException(status_code=404, detail="ML set not found")
    take_ids = [str(item).strip() for item in payload.take_ids if str(item).strip()]
    if not take_ids:
        raise HTTPException(status_code=400, detail="No take_ids provided")
    memberships = {
        str(item.get("take_id") or ""): item
        for item in service.list_ml_set_memberships(dataset_id=dataset_id, ml_set_id=ml_set_id)
        if str(item.get("take_id") or "")
    }
    updated = 0
    skipped: list[str] = []
    for take_id in take_ids:
        current = memberships.get(take_id)
        if not current:
            skipped.append(take_id)
            continue
        next_split = payload.split if payload.split is not None else str(current.get("split") or "unassigned")
        if next_split not in {"train", "validation", "test", "holdout", "calibration", "unassigned"}:
            raise HTTPException(status_code=400, detail=f"Invalid split: {next_split}")
        service.add_take_to_ml_set(
            dataset_id=dataset_id,
            ml_set_id=ml_set_id,
            take_id=take_id,
            split=next_split,
            physical_object_id=str(current.get("physical_object_id") or "") or None,
            include=payload.include if payload.include is not None else bool(current.get("include", True)),
            default_trainable=payload.default_trainable if payload.default_trainable is not None else current.get("default_trainable"),
            trainable=current.get("trainable"),
            notes=str(current.get("notes") or "") or None,
            expected_label=str(current.get("expected_label") or "") or None,
            expected_class=payload.expected_class if payload.expected_class is not None else (str(current.get("expected_class") or "") or None),
            expected_subclass=payload.expected_subclass if payload.expected_subclass is not None else (str(current.get("expected_subclass") or "") or None),
            measurements_mm=current.get("measurements_mm") if isinstance(current.get("measurements_mm"), dict) else {},
            raw_label=str(current.get("raw_label") or "") or None,
            label_policy=str(current.get("label_policy") or "") or None,
            review_required=payload.review_required if payload.review_required is not None else current.get("review_required"),
            normalization_version=str(current.get("normalization_version") or "") or None,
            source_row=str(current.get("source_row") or "") or None,
            extra_fields=current.get("extra_fields") if isinstance(current.get("extra_fields"), dict) else {},
        )
        updated += 1
    return {"ok": True, "updated_count": updated, "skipped_count": len(skipped), "skipped_ids": skipped[:100]}


@app.get("/api/ml-sets/{ml_set_id}/members")
def ml_set_members(
    ml_set_id: str,
    dataset_id: str | None = None,
    physical_object_id: str | None = None,
    raw_label: str | None = None,
    normalized_class: str | None = None,
    superclass: str | None = None,
    membership_status: str | None = None,
    split: str | None = None,
    session_id: str | None = None,
    search: str | None = None,
    validation_status: str | None = None,
    limit: int = 200,
    offset: int = 0,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return MLSetSummaryService(settings).list_members(
            ml_set_id,
            dataset_id,
            physical_object_id=physical_object_id,
            raw_label=raw_label,
            normalized_class=normalized_class,
            superclass=superclass,
            membership_status=membership_status,
            split=split,
            session_id=session_id,
            search=search,
            validation_status=validation_status,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/ml-sets/{ml_set_id}/facets")
def dataset_ml_set_member_facets(
    dataset_id: str,
    ml_set_id: str,
    physical_object_id: str | None = None,
    raw_label: str | None = None,
    normalized_class: str | None = None,
    superclass: str | None = None,
    membership_status: str | None = None,
    split: str | None = None,
    session_id: str | None = None,
    search: str | None = None,
    validation_status: str | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return MLSetSummaryService(settings).build_member_facets(
            ml_set_id,
            dataset_id,
            physical_object_id=physical_object_id,
            raw_label=raw_label,
            normalized_class=normalized_class,
            superclass=superclass,
            membership_status=membership_status,
            split=split,
            session_id=session_id,
            search=search,
            validation_status=validation_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/ml-sets/{ml_set_id}/summary")
def ml_set_summary(ml_set_id: str, dataset_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLSetSummaryService(settings).build_summary(ml_set_id, dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/ml-sets/{ml_set_id}/class-distribution")
def ml_set_class_distribution(ml_set_id: str, dataset_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = ml_set_summary(ml_set_id, dataset_id=dataset_id, settings=settings)
    return {
        "class_distribution": payload.get("class_distribution") or {},
        "superclass_distribution": payload.get("superclass_distribution") or {},
        "split_by_class": payload.get("split_by_class") or {},
    }


@app.get("/api/ml-sets/{ml_set_id}/split-summary")
def ml_set_split_summary(ml_set_id: str, dataset_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = ml_set_summary(ml_set_id, dataset_id=dataset_id, settings=settings)
    return payload.get("split_summary") or {}


@app.get("/api/ml-sets/{ml_set_id}/warnings")
def ml_set_warnings(ml_set_id: str, dataset_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = ml_set_summary(ml_set_id, dataset_id=dataset_id, settings=settings)
    return {"warnings": payload.get("warnings") or []}


@app.get("/api/ml-sets/{ml_set_id}/representative-samples")
def ml_set_representative_samples(ml_set_id: str, dataset_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    payload = ml_set_summary(ml_set_id, dataset_id=dataset_id, settings=settings)
    return {"representative_samples": payload.get("representative_samples") or {}}


@app.get("/api/ml-sets/{ml_set_id}/export")
def ml_set_export(ml_set_id: str, kind: str, dataset_id: str | None = None, settings: ApiSettings = Depends(get_settings)) -> FileResponse:
    try:
        path = MLSetSummaryService(settings).export_file(ml_set_id, kind, dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Export not found: {kind}")
    return FileResponse(path)


@app.post("/api/ml-ingestion/runs")
def create_ingestion_run(payload: IngestionRunCreateRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.create_run(dataset_id=payload.dataset_id, session_ids=payload.session_ids, name=payload.name)


@app.get("/api/ml-ingestion/runs/{run_id}")
def get_ingestion_run(run_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.get_run(run_id)


@app.post("/api/ml-ingestion/runs/{run_id}/table")
async def ingest_operator_table(
    run_id: str,
    request: Request,
    file: UploadFile | None = File(default=None),
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            if file is None:
                raise ValueError("No file uploaded.")
            data = await file.read()
            return service.ingest_uploaded_file(run_id, filename=file.filename or "uploaded_table", content_bytes=data)
        payload_raw = await request.json()
        payload = IngestionTableRequest(**payload_raw)
        if payload is None:
            raise ValueError("Missing ingestion payload.")
        return service.ingest_table(run_id, content=payload.content, input_format=payload.input_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ml-ingestion/runs/{run_id}/reconcile")
def reconcile_ingestion_run(run_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.reconcile(run_id)


@app.post("/api/ml-ingestion/runs/{run_id}/policy")
def update_ingestion_policy(run_id: str, payload: IngestionPolicyUpdateRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.configure_policy(run_id, payload.policy)


@app.post("/api/ml-ingestion/runs/{run_id}/manifest")
def generate_ingestion_manifest(run_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.generate_manifest(run_id)


@app.post("/api/ml-ingestion/runs/{run_id}/materialize")
def materialize_ingestion_run(run_id: str, payload: IngestionMaterializeRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.materialize(run_id, payload.ml_set_id)


@app.post("/api/ml-ingestion/runs/{run_id}/apply-metadata")
def apply_ingestion_labels_to_metadata(run_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = IngestionWizardService(settings.data_dir)
    return service.apply_labels_to_metadata(run_id)


@app.put("/api/datasets/{dataset_id}/sessions/{session_id}", response_model=DatasetSessionSummary)
def update_dataset_session(dataset_id: str, session_id: str, payload: DatasetSessionRequest, settings: ApiSettings = Depends(get_settings)) -> DatasetSessionSummary:
    service = DatasetService(settings.data_dir)
    updated = service.update_session(dataset_id, session_id, payload.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found")
    take_count = _dataset_session_take_count(settings, dataset_id=dataset_id, session_id=session_id)
    return DatasetSessionSummary(**updated, take_count=take_count)


@app.get("/api/feature-analytics/features")
def feature_analytics_features(
    request: Request = None,
    query: dict[str, Any] = Depends(parse_feature_analytics_query),
    max_takes: int = 600,
    time_budget_ms: int = 2500,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    started = time.monotonic()
    resolved_query = query if isinstance(query, dict) else (parse_feature_analytics_query(request) if request is not None else {})
    logger.info(
        "feature_analytics.features:start dataset=%s session=%s pipeline=%s max_takes=%s budget_ms=%s query=%s",
        resolved_query.get("dataset_id"),
        resolved_query.get("session_id"),
        resolved_query.get("pipeline_id"),
        max_takes,
        time_budget_ms,
        str(request.url) if request is not None else "<direct-call>",
    )
    records, debug = build_feature_records(settings, query=resolved_query, max_takes=max_takes, time_budget_ms=time_budget_ms)
    scope_summary = summarize_feature_scope(records)
    payload = {
        "record_count": scope_summary["record_count"],
        "object_count": scope_summary["object_count"],
        "scope_summary": scope_summary,
        "feature_definitions": _feature_definitions(records),
        "debug": debug,
    }
    logger.info("feature_analytics.features:done records=%s takes_scanned=%s duration_ms=%.2f", payload["record_count"], debug.get("takes_scanned"), (time.monotonic() - started) * 1000.0)
    return payload


@app.get("/api/feature-analytics/distributions")
def feature_analytics_distributions(
    feature_key: str,
    request: Request = None,
    group_by: str = "superclass",
    bins: int = 24,
    mode: str = "count",
    query: dict[str, Any] = Depends(parse_feature_analytics_query),
    max_takes: int = 600,
    time_budget_ms: int = 2500,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    started = time.monotonic()
    resolved_query = query if isinstance(query, dict) else (parse_feature_analytics_query(request) if request is not None else {})
    logger.info("feature_analytics.distributions:start feature=%s group_by=%s bins=%s mode=%s query=%s", feature_key, group_by, bins, mode, str(request.url) if request is not None else "<direct-call>")
    records, debug = build_feature_records(settings, query=resolved_query, max_takes=max_takes, time_budget_ms=time_budget_ms)
    resolved_bins = max(4, min(int(bins), 128))
    resolved_mode = "density" if mode == "density" else "count"
    resolved_group = group_by if group_by in {"label", "raw_label", "normalized_class", "physical_object_id", "processed_superclass", "labeled_superclass", "processed_class"} else "processed_superclass"
    payload = build_distributions(records, feature_key=feature_key, group_by=resolved_group, bins=resolved_bins, mode=resolved_mode)
    payload["debug"] = debug
    logger.info("feature_analytics.distributions:done groups=%s records=%s duration_ms=%.2f", len(payload.get("groups") or []), len(records), (time.monotonic() - started) * 1000.0)
    return payload


@app.get("/api/feature-analytics/scatter")
def feature_analytics_scatter(
    x_feature_key: str,
    y_feature_key: str,
    request: Request = None,
    group_by: str = "superclass",
    limit: int = 2000,
    query: dict[str, Any] = Depends(parse_feature_analytics_query),
    max_takes: int = 600,
    time_budget_ms: int = 2500,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    started = time.monotonic()
    resolved_query = query if isinstance(query, dict) else (parse_feature_analytics_query(request) if request is not None else {})
    logger.info("feature_analytics.scatter:start x=%s y=%s group_by=%s query=%s", x_feature_key, y_feature_key, group_by, str(request.url) if request is not None else "<direct-call>")
    records, debug = build_feature_records(settings, query=resolved_query, max_takes=max_takes, time_budget_ms=time_budget_ms)
    rows: list[dict[str, Any]] = []
    for item in records:
        x = (item.get("features") or {}).get(x_feature_key)
        y = (item.get("features") or {}).get(y_feature_key)
        if x is None or y is None:
            continue
        rows.append(
            {
                "take_id": item.get("take_id"),
                "object_id": item.get("object_id"),
                "x": x,
                "y": y,
                "group": (
                    ((item.get("raw_labels") or ["UNKNOWN"])[0] if group_by == "raw_label" else None)
                    or (item.get("processed_superclass") if group_by == "processed_superclass" else None)
                    or (item.get("labeled_superclass") if group_by == "labeled_superclass" else None)
                    or (item.get("processed_class") if group_by == "processed_class" else None)
                    or (item.get("normalized_class") if group_by == "normalized_class" else None)
                    or (((item.get("labels") or ["UNKNOWN"])[0]) if group_by == "label" else None)
                    or (item.get("physical_object_id") if group_by == "physical_object_id" else None)
                    or item.get("superclass")
                    or "UNKNOWN"
                ),
            }
        )
        if len(rows) >= max(10, min(int(limit), 10000)):
            break
    payload = {"x_feature_key": x_feature_key, "y_feature_key": y_feature_key, "group_by": group_by, "points": rows, "debug": debug}
    logger.info("feature_analytics.scatter:done points=%s duration_ms=%.2f", len(rows), (time.monotonic() - started) * 1000.0)
    return payload


@app.get("/api/feature-analytics/objects")
def feature_analytics_objects(
    feature_key: str,
    request: Request = None,
    min_value: float | None = None,
    max_value: float | None = None,
    limit: int = 200,
    query: dict[str, Any] = Depends(parse_feature_analytics_query),
    max_takes: int = 600,
    time_budget_ms: int = 2500,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    started = time.monotonic()
    resolved_query = query if isinstance(query, dict) else (parse_feature_analytics_query(request) if request is not None else {})
    logger.info("feature_analytics.objects:start feature=%s min=%s max=%s limit=%s query=%s", feature_key, min_value, max_value, limit, str(request.url) if request is not None else "<direct-call>")
    records, debug = build_feature_records(settings, query=resolved_query, max_takes=max_takes, time_budget_ms=time_budget_ms)
    objects = filter_records_by_feature_range(
        records,
        feature_key=feature_key,
        min_value=min_value,
        max_value=max_value,
        limit=limit,
    )
    scope_summary = summarize_records_by_feature_range(
        records,
        feature_key=feature_key,
        min_value=min_value,
        max_value=max_value,
    )
    payload = {
        "feature_key": feature_key,
        "min_value": min_value,
        "max_value": max_value,
        "limit": limit,
        "objects": objects,
        "scope_summary": scope_summary,
        "stats": summarize_feature_values(records, feature_key),
        "debug": debug,
    }
    logger.info("feature_analytics.objects:done objects=%s duration_ms=%.2f", len(objects), (time.monotonic() - started) * 1000.0)
    return payload


@app.get("/api/feature-analytics/filter-options")
def feature_analytics_filter_options(
    request: Request = None,
    query: dict[str, Any] = Depends(parse_feature_analytics_query),
    max_takes: int = 600,
    time_budget_ms: int = 2500,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    started = time.monotonic()
    resolved_query = query if isinstance(query, dict) else (parse_feature_analytics_query(request) if request is not None else {})
    logger.info("feature_analytics.filter_options:start query=%s", str(request.url) if request is not None else "<direct-call>")
    records, debug = build_feature_records(settings, query=resolved_query, max_takes=max_takes, time_budget_ms=time_budget_ms)
    options = build_filter_options(records)
    options["debug"] = debug
    logger.info("feature_analytics.filter_options:done records=%s duration_ms=%.2f", len(records), (time.monotonic() - started) * 1000.0)
    return options


@app.get("/api/sessions/{session_id}/summary")
def session_summary_endpoint(session_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    summary = get_session_summary(settings, session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


@app.get("/api/dataset-sessions/{session_id}/summary")
def dataset_session_summary_endpoint(
    session_id: str,
    dataset_id: str | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    service = DatasetSessionExportService(settings)
    try:
        return service.build_session_summary(session_id=session_id, dataset_id=dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/dataset-sessions/{session_id}/export")
def dataset_session_export_endpoint(
    session_id: str,
    dataset_id: str | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> Response:
    service = DatasetSessionExportService(settings)
    try:
        payload = service.export_session(session_id=session_id, dataset_id=dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    headers = {
        "Content-Disposition": f'attachment; filename="dataset_session_{safe_session}.json"'
    }
    return Response(content=encoded, media_type="application/json", headers=headers)


@app.get("/api/takes/{take_id}", response_model=TakeDetail)
def take_detail(take_id: str, settings: ApiSettings = Depends(get_settings), run_id: str | None = None) -> TakeDetail:
    detail = get_take_detail(settings, take_id, run_id=run_id)
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


def _resolve_bulk_take_ids(mode: str, take_ids: list[str], filters: TakeBulkFilters | None, settings: ApiSettings) -> list[str]:
    if mode == "ids":
        return [str(item).strip() for item in take_ids if str(item).strip()]
    if mode != "filter":
        raise HTTPException(status_code=400, detail="Invalid mode")
    f = filters or TakeBulkFilters()
    if f.ml_set_id:
        try:
            payload = MLSetSummaryService(settings).list_members(
                f.ml_set_id,
                f.dataset_id,
                physical_object_id=f.physical_object_id,
                raw_label=f.raw_label,
                normalized_class=f.normalized_class,
                superclass=f.superclass,
                membership_status=f.membership_status,
                split=f.split,
                session_id=f.session_id,
                search=f.search,
                validation_status=f.validation_status,
                limit=500,
                offset=0,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        candidate_ids: list[str] = [str(item.get("take_id") or "").strip() for item in payload.get("items") or [] if str(item.get("take_id") or "").strip()]
        while payload.get("has_more"):
            payload = MLSetSummaryService(settings).list_members(
                f.ml_set_id,
                f.dataset_id,
                physical_object_id=f.physical_object_id,
                raw_label=f.raw_label,
                normalized_class=f.normalized_class,
                superclass=f.superclass,
                membership_status=f.membership_status,
                split=f.split,
                session_id=f.session_id,
                search=f.search,
                validation_status=f.validation_status,
                limit=500,
                offset=int(payload.get("next_offset") or 0),
            )
            candidate_ids.extend([str(item.get("take_id") or "").strip() for item in payload.get("items") or [] if str(item.get("take_id") or "").strip()])
        return candidate_ids
    candidate_ids: list[str] = []
    page_offset = 0
    while True:
        page = list_takes_paged(
            settings,
            limit=200,
            offset=page_offset,
            session_id=f.session_id,
            dataset_id=f.dataset_id,
            validation_status=f.validation_status,
            tag=f.tag,
            search=f.search,
            created_from=f.created_from,
            created_to=f.created_to,
            expected_class=f.expected_class,
            split=f.split,
            calibration_linkage_only=bool(f.calibration_linkage_only),
            physical_object_id=f.physical_object_id,
            superclass_label=f.superclass,
            include_archived=False,
        )
        items = page.get("items") or []
        candidate_ids.extend([str(item.take_id) for item in items if getattr(item, "take_id", None)])
        if not page.get("has_more"):
            break
        page_offset = int(page.get("next_offset") or (page_offset + len(items)))
    return candidate_ids


@app.post("/api/takes/bulk-metadata")
def bulk_update_take_metadata(payload: TakeBulkMetadataRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    allowed_modes = {"ids", "filter"}
    allowed_actions = {
        "move_to_dataset",
        "move_to_session",
        "ml_set_add",
        "ml_set_remove",
        "tag_add",
        "tag_remove",
        "tag_replace",
        "set_validation_status",
        "set_split",
        "set_expected_class",
    }
    if payload.mode not in allowed_modes:
        raise HTTPException(status_code=400, detail="Invalid mode")
    if payload.action not in allowed_actions:
        raise HTTPException(status_code=400, detail="Invalid action")

    candidate_ids = _resolve_bulk_take_ids(payload.mode, payload.take_ids, payload.filters, settings)

    excluded = {str(item).strip() for item in payload.exclude_take_ids if str(item).strip()}
    target_ids = [take_id for take_id in candidate_ids if take_id and take_id not in excluded]

    if payload.action == "move_to_session" and not (payload.session_id or "").strip():
        raise HTTPException(status_code=400, detail="session_id is required for move_to_session")
    if payload.action == "move_to_dataset" and not (payload.dataset_id or "").strip():
        raise HTTPException(status_code=400, detail="dataset_id is required for move_to_dataset")

    if payload.dry_run:
        return {
            "ok": True,
            "mode": payload.mode,
            "action": payload.action,
            "matched_count": len(candidate_ids),
            "affected_count": len(target_ids),
            "skipped_count": max(0, len(candidate_ids) - len(target_ids)),
            "failed_count": 0,
            "failed_ids": [],
        }

    service = DatasetService(settings.data_dir)
    failed_ids: list[str] = []
    affected_count = 0
    skipped_count = 0

    for take_id in target_ids:
        metadata_path = settings.incoming_dir / take_id / "metadata.json"
        if not metadata_path.is_file():
            skipped_count += 1
            continue
        source_meta = read_operations_json(metadata_path)
        source_meta = source_meta if isinstance(source_meta, dict) else {}
        current = service.load_take_metadata(take_id=take_id, source_metadata=source_meta)
        dataset_id = str(current.get("dataset_id") or source_meta.get("dataset_id") or payload.dataset_id or "legacy")
        session_id = str(current.get("session_id") or source_meta.get("session_id") or payload.session_id or "legacy")
        if payload.action == "move_to_dataset":
            dataset_id = str(payload.dataset_id or dataset_id or "legacy")
            session_id = str(payload.session_id or session_id or "legacy")
        elif payload.action == "move_to_session":
            session_id = str(payload.session_id or session_id or "legacy")
            dataset_id = str(payload.dataset_id or dataset_id or "legacy")
        if service.get_dataset(dataset_id) is None:
            service.create_dataset(dataset_id=dataset_id, name=dataset_id, description="Auto-generated dataset")
        if service.get_session(dataset_id, session_id) is None:
            service.create_session(dataset_id=dataset_id, session_id=session_id, name=session_id, description="Auto-generated session")

        updates: dict[str, Any] = {}
        if payload.action == "move_to_dataset":
            updates["dataset_id"] = dataset_id
            updates["session_id"] = session_id
        elif payload.action == "move_to_session":
            updates["session_id"] = session_id
            updates["dataset_id"] = dataset_id
        elif payload.action == "tag_add":
            existing_tags = [str(item) for item in (current.get("tags") or [])]
            updates["tags"] = list(dict.fromkeys(existing_tags + [str(item) for item in (payload.tags or []) if str(item).strip()]))
        elif payload.action == "ml_set_add":
            existing_sets = [str(item) for item in (current.get("ml_set_ids") or []) if str(item).strip()]
            updates["ml_set_ids"] = list(dict.fromkeys(existing_sets + [str(item) for item in (payload.tags or []) if str(item).strip()]))
        elif payload.action == "ml_set_remove":
            remove_sets = {str(item).strip() for item in (payload.tags or []) if str(item).strip()}
            updates["ml_set_ids"] = [str(item) for item in (current.get("ml_set_ids") or []) if str(item).strip() not in remove_sets]
        elif payload.action == "tag_remove":
            remove = {str(item).strip() for item in (payload.tags or []) if str(item).strip()}
            updates["tags"] = [str(item) for item in (current.get("tags") or []) if str(item) not in remove]
        elif payload.action == "tag_replace":
            updates["tags"] = [str(item).strip() for item in (payload.tags or []) if str(item).strip()]
        elif payload.action == "set_validation_status":
            updates["validation_status"] = payload.validation_status
        elif payload.action == "set_split":
            updates["split"] = payload.split
        elif payload.action == "set_expected_class":
            updates["expected_class"] = payload.expected_class

        try:
            service.upsert_take_metadata(
                take_id=take_id,
                dataset_id=dataset_id,
                session_id=session_id,
                updates=updates,
                source_metadata=source_meta,
            )
            affected_count += 1
        except Exception:
            failed_ids.append(take_id)

    failed_count = len(failed_ids)
    return {
        "ok": True,
        "mode": payload.mode,
        "action": payload.action,
        "matched_count": len(candidate_ids),
        "affected_count": affected_count,
        "skipped_count": skipped_count + max(0, len(candidate_ids) - len(target_ids)),
        "failed_count": failed_count,
        "failed_ids": failed_ids[:100],
    }


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

    selected_recipe = None
    if payload.recipe_id:
        try:
            selected_recipe = RecipeService(settings).recipe_for_run(pipeline_id, payload.recipe_id, version=payload.recipe_version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    merged_stage_params: dict[str, Any] | None = None
    if selected_recipe is not None or payload.stage_params:
        merged_stage_params = {}
        if selected_recipe and isinstance(selected_recipe.get("stage_params"), dict):
            merged_stage_params.update(selected_recipe["stage_params"])
        if payload.stage_params:
            merged_stage_params.update(payload.stage_params)

    try:
        response = dispatch_take_processing(
            settings=settings,
            take_id=take_id,
            pipeline_id=pipeline_id,
            reprocess=payload.reprocess,
            source_id=source_id or None,
            recipe_version_id=str(recipe_version_id) if recipe_version_id else None,
            recipe_metadata=selected_recipe,
            acquisition_group_id=str(resolved_acquisition_group_id) if resolved_acquisition_group_id else None,
            calibration_profile_id=str(calibration_profile_id) if calibration_profile_id else None,
            stage_params=merged_stage_params or None,
        )
        if binding_warning:
            response["warning"] = binding_warning
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/processing-jobs")
def create_processing_job(payload: ProcessingJobCreateRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        job = get_processing_job_service(settings).create_job(payload)
        return {"job": job.model_dump(mode="json")}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/processing-jobs")
def list_processing_jobs(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    service = get_processing_job_service(settings)
    jobs = [job.model_dump(mode="json") for job in service.list_jobs()]
    counts = {
        "queued": sum(1 for job in jobs if str(job.get("status")) == "queued"),
        "running": sum(1 for job in jobs if str(job.get("status")) == "running"),
        "completed": sum(1 for job in jobs if str(job.get("status")) == "completed"),
        "failed": sum(1 for job in jobs if str(job.get("status")) == "failed"),
        "cancelled": sum(1 for job in jobs if str(job.get("status")) == "cancelled"),
        "partial": sum(1 for job in jobs if str(job.get("status")) == "partial"),
    }
    return {"jobs": jobs, "counts": counts}


@app.get("/api/processing-jobs/{job_id}")
def get_processing_job(job_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    job = get_processing_job_service(settings).load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown processing job: {job_id}")
    return {"job": job.model_dump(mode="json")}


@app.post("/api/processing-jobs/{job_id}/cancel")
def cancel_processing_job(job_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        job = get_processing_job_service(settings).cancel_job(job_id)
        payload = ProcessingJobCancelResponse(ok=True, job_id=job_id, status=job.status)
        return payload.model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/ml/feature-jobs")
def create_feature_job(payload: FeatureJobRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        job = get_feature_job_service(settings).create_job(payload)
        return {"job_id": job.id, "status": job.status, "job": job.model_dump(mode="json")}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/feature-jobs/{job_id}")
def get_feature_job(job_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    job = get_feature_job_service(settings).load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown feature job: {job_id}")
    return {"job_id": job.id, "status": job.status, "job": job.model_dump(mode="json")}


@app.post("/api/ml/feature-jobs/{job_id}/cancel")
def cancel_feature_job(job_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        job = get_feature_job_service(settings).cancel_job(job_id)
        return {"ok": True, "job_id": job_id, "status": job.status}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


@app.get("/api/ml/datasets")
def ml_datasets(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return MLService(settings.data_dir).list_datasets()


@app.post("/api/ml/datasets")
def ml_create_dataset(payload: MLDatasetRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return MLService(settings.data_dir).create_dataset(payload.model_dump(mode="json"))


@app.get("/api/ml/experiments")
def ml_experiments(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return MLService(settings.data_dir).list_experiments()


@app.post("/api/ml/experiments")
def ml_run_experiment(payload: MLExperimentRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).run_experiment(payload.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/models")
def ml_models(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return MLService(settings.data_dir).list_models()


@app.get("/api/ml/backends")
def ml_backends(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return MLService(settings.data_dir).list_backends()


@app.post("/api/ml/models/{model_id}/promote")
def ml_promote_model(model_id: str, payload: PromoteModelRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).promote_model(model_id=model_id, notes=payload.notes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ml/evaluate")
def ml_evaluate(payload: MLEvaluateRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).run_experiment(payload.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/deployments")
def ml_deployments(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return MLService(settings.data_dir).list_deployments()


@app.post("/api/ml/deployments")
def ml_create_deployment(payload: MLDeploymentRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).deploy_model(payload.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ml/deployments/{deployment_id}/activate")
def ml_activate_deployment(deployment_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).activate_deployment(deployment_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ml/deployments/{deployment_id}/deactivate")
def ml_deactivate_deployment(deployment_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).deactivate_deployment(deployment_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ml/deployments/{deployment_id}/rollback")
def ml_rollback_deployment(deployment_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).rollback_deployment(deployment_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/deployments/active")
def ml_active_deployments(settings: ApiSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    return MLService(settings.data_dir).list_active_deployments()


@app.post("/api/ml/smoke-test/run")
def ml_run_smoke_test(payload: MLSmokeRunRequest, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/ml/run_ml_smoke_test.py",
        "--data-dir",
        str(settings.data_dir),
        "--ci-mode" if payload.ci_mode else "",
        "--force-invalid-model" if payload.force_invalid_model else "",
    ]
    cmd = [item for item in cmd if item]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(settings.data_dir.parent), check=False)
    latest = settings.data_dir / "ml" / "smoke_test_results" / "latest.json"
    result = _read_json(latest) if latest.is_file() else None
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "result": result,
    }


@app.get("/api/ml/smoke-test/latest")
def ml_smoke_test_latest(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    latest = settings.data_dir / "ml" / "smoke_test_results" / "latest.json"
    payload = _read_json(latest)
    return payload or {"passed": False, "phases": [], "failures": [{"name": "smoke-test", "detail": "no results found"}]}


@app.get("/api/ml/smoke-test/history")
def ml_smoke_test_history(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    history_dir = settings.data_dir / "ml" / "smoke_test_results" / "history"
    rows: list[dict[str, Any]] = []
    if history_dir.is_dir():
        for path in sorted(history_dir.glob("smoke_test_*.json"), reverse=True):
            payload = _read_json(path)
            if isinstance(payload, dict):
                rows.append(payload)
    return {"items": rows}


@app.get("/api/ml/runtime-health")
def ml_runtime_health(settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    day_key = datetime.now(UTC).strftime("%Y%m%d")
    stats_path = settings.data_dir / "ml" / "runtime_stats" / f"{day_key}.json"
    stats = _read_json(stats_path) if stats_path.is_file() else {}
    inf = int((stats or {}).get("inference_count", 0))
    fallback = int((stats or {}).get("fallback_count", 0))
    heuristic = int((stats or {}).get("heuristic_usage_count", 0))
    incompatible = int((stats or {}).get("incompatible_deployment_count", 0))
    latest_path = settings.data_dir / "ml" / "smoke_test_results" / "latest.json"
    latest = _read_json(latest_path) if latest_path.is_file() else None
    return {
        "runtime_stats": stats or {},
        "totals": {
            "inference_count": inf,
            "fallback_rate": (float(fallback) / float(inf)) if inf else 0.0,
            "heuristic_usage_rate": (float(heuristic) / float(inf)) if inf else 0.0,
            "incompatible_deployment_count": incompatible,
            "average_inference_time_ms": float((stats or {}).get("average_inference_time_ms", 0.0)),
        },
        "backend_breakdown": {
            "inference_by_backend": (stats or {}).get("inference_by_backend", {}),
            "fallback_by_backend": (stats or {}).get("fallback_by_backend", {}),
            "latency_by_backend_ms": (stats or {}).get("latency_by_backend_ms", {}),
        },
        "latest_smoke_test": latest,
    }


@app.get("/api/ml/runtime-health/history")
def ml_runtime_health_history(days: int = 14, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return MLService(settings.data_dir).runtime_health_history(days=days)


@app.get("/api/ml/datasets/{dataset_id}/insights")
def ml_dataset_insights(dataset_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).dataset_insights(dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/datasets/{dataset_id}/labels-preview")
def ml_labels_preview(dataset_id: str, limit: int = 200, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).labels_preview(dataset_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/datasets/{dataset_id}/features-preview")
def ml_features_preview(dataset_id: str, limit: int = 200, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).features_preview(dataset_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/datasets/{dataset_id}/split-preview")
def ml_split_preview(dataset_id: str, strategy: str = "by_session", seed: int = 42, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    try:
        return MLService(settings.data_dir).split_preview(dataset_id, strategy=strategy, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ml/experiments/{experiment_id}/logs")
def ml_experiment_logs(experiment_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return MLService(settings.data_dir).training_logs(experiment_id)


@app.get("/api/ml/experiments/{experiment_id}/evaluation-preview")
def ml_experiment_evaluation_preview(experiment_id: str, settings: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
    return MLService(settings.data_dir).evaluation_preview(experiment_id)


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
    if filename in {
        "height_above_belt.values.f32",
        "normalized_heightmap.values.f32",
        "raw_sensor_z.values.f32",
        "raw_heightmap.values.f32",
        "valid_mask.u8",
        "normalized_heightmap.render_context.json",
        "normalized_heightmap_display.json",
        "normalized_heightmap.png",
    }:
        size_bytes = path.stat().st_size if path.exists() else -1
        if filename.endswith(".values.f32"):
            sample_count = size_bytes // 4 if size_bytes >= 0 else -1
            detail = f"float32_count={sample_count}"
        elif filename.endswith(".u8"):
            detail = f"uint8_count={size_bytes}"
        else:
            detail = f"bytes={size_bytes}"
        print(
            "[api-hover-debug] serving hover/render file "
            f"take_id={take_id} filename={filename} path={path} {detail}",
            flush=True,
        )
    return FileResponse(path)


@app.get("/api/comparisons/{comparison_id}/files/{filename:path}")
def comparison_file(
    comparison_id: str,
    filename: str,
    settings: ApiSettings = Depends(get_settings),
) -> FileResponse:
    path = resolve_comparison_file(settings, comparison_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


def _object_bbox_from_detail(detail: TakeDetail, object_id: str) -> tuple[float, float, float, float] | None:
    result = detail.result if isinstance(detail.result, dict) else {}
    oid_str = str(object_id).strip()

    def _bbox_from_row(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
        direct = row.get("bbox_px")
        if isinstance(direct, list) and len(direct) >= 4:
            try:
                x, y, w, h = float(direct[0]), float(direct[1]), float(direct[2]), float(direct[3])
                if w > 0 and h > 0:
                    return (x, y, w, h)
            except Exception:
                pass
        prov = row.get("measurement_provenance")
        if isinstance(prov, dict):
            for key in ("segmented_bbox_px", "raw_bbox_px"):
                item = prov.get(key)
                if isinstance(item, list) and len(item) >= 4:
                    try:
                        x, y, w, h = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                        if w > 0 and h > 0:
                            return (x, y, w, h)
                    except Exception:
                        continue
        geometry = row.get("geometry")
        if isinstance(geometry, dict):
            box = geometry.get("bounding_box")
            if isinstance(box, dict):
                try:
                    x = float(box.get("x") or 0.0)
                    y = float(box.get("y") or 0.0)
                    w = float(box.get("width") or 0.0)
                    h = float(box.get("height") or 0.0)
                    if w > 0 and h > 0:
                        return (x, y, w, h)
                except Exception:
                    pass
        return None

    merged_objects = []
    merged_objects.extend(item for item in (result.get("objects") or []) if isinstance(item, dict))
    merged_objects.extend(item for item in (result.get("rejected_objects") or []) if isinstance(item, dict))
    for row in merged_objects:
        if str(row.get("object_id") or "").strip() == oid_str:
            bbox = _bbox_from_row(row)
            if bbox is not None:
                return bbox
            source_candidate_id = str(row.get("source_candidate_id") or "").strip()
            if source_candidate_id:
                candidates = result.get("object_candidates") if isinstance(result.get("object_candidates"), list) else []
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    if str(candidate.get("id") or candidate.get("candidate_id") or "").strip() != source_candidate_id:
                        continue
                    bbox_candidate = candidate.get("bbox_px")
                    if isinstance(bbox_candidate, list) and len(bbox_candidate) >= 4:
                        try:
                            x, y, w, h = float(bbox_candidate[0]), float(bbox_candidate[1]), float(bbox_candidate[2]), float(bbox_candidate[3])
                            if w > 0 and h > 0:
                                return (x, y, w, h)
                        except Exception:
                            continue
    return None


def _stage_crop_artifact_path(settings: ApiSettings, detail: TakeDetail, object_id: str) -> Path | None:
    result = detail.result if isinstance(detail.result, dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), list) else []
    target_id = f"object_crop_{str(object_id).strip()}"
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_id") or "") != target_id:
            continue
        candidate_path = artifact.get("path")
        if not isinstance(candidate_path, str) or not candidate_path.strip():
            continue
        return safe_take_file(settings, detail.take_id, candidate_path.strip())
    return None


def _thumbnail_source_path(settings: ApiSettings, detail: TakeDetail, mode: str) -> Path | None:
    resolved_mode = str(mode or "auto").strip().lower()
    groups = detail.assets or {}
    if resolved_mode == "classification_overlay":
        result = detail.result if isinstance(detail.result, dict) else {}
        artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), list) else []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_id = str(artifact.get("artifact_id") or "").strip()
            candidate_path = str(artifact.get("path") or "").strip()
            if artifact_id != "classification_overlay_image" or not candidate_path:
                continue
            candidate = safe_take_file(settings, detail.take_id, candidate_path)
            if candidate is not None and candidate.is_file():
                return candidate
    preferred_groups = {
        "heightmap": (("heightmap", ["heightmap", "height"]),),
        "reflectance": (("reflectance", ["reflectance"]), ("rgb", ["rgb"]), ("laser_rgb", ["laser_rgb"])),
    }.get(resolved_mode)
    if preferred_groups is not None:
        for group, keys in preferred_groups:
            payload = groups.get(group) or {}
            for key in keys:
                filename = payload.get(key)
                if not filename:
                    continue
                candidate = settings.incoming_dir / detail.take_id / str(filename)
                if candidate.is_file():
                    return candidate
    source_path, _resolved = resolve_source_image_path(settings, detail)
    return source_path


def _neutral_placeholder(cache_path: Path, size_px: int) -> Path:
    if cache_path.exists():
        return cache_path
    image = Image.new("RGB", (size_px, size_px), color=(226, 232, 240))
    image.save(cache_path, format="JPEG", quality=82, optimize=True)
    return cache_path


def _thumbnail_artifact_candidate(detail: TakeDetail, *artifact_ids: str) -> tuple[str | None, str | None]:
    result = detail.result if isinstance(detail.result, dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), list) else []
    wanted = {item for item in artifact_ids if item}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "").strip()
        if artifact_id not in wanted:
            continue
        candidate_path = str(artifact.get("path") or "").strip()
        if candidate_path:
            return artifact_id, candidate_path
    return None, None


def _build_thumbnail_plan(
    settings: ApiSettings,
    detail: TakeDetail,
    object_id: str,
    *,
    mode: str,
    semantic_group: str | None = None,
) -> dict[str, Any]:
    requested_mode = str(mode or "auto").strip().lower()
    if requested_mode not in {"auto", "heightmap", "reflectance", "classification_overlay"}:
        requested_mode = "auto"
    semantic = str(semantic_group or "").strip().lower()
    bbox = _object_bbox_from_detail(detail, object_id)
    fallback_reason: str | None = None
    fallback_used = False
    resolved_source = "placeholder"
    resolved_artifact_id: str | None = None
    resolved_artifact_path: str | None = None
    colorized = False
    source_path: Path | None = None

    def resolve_candidate(candidate_mode: str) -> tuple[Path | None, str, str | None, str | None, bool]:
        if candidate_mode == "classification_overlay":
            artifact_id, artifact_path = _thumbnail_artifact_candidate(detail, "classification_overlay_image")
            if artifact_path:
                path = safe_take_file(settings, detail.take_id, artifact_path)
                if path is not None and path.is_file():
                    return path, "classification_overlay", artifact_id, artifact_path, True
            return None, "classification_overlay", artifact_id, artifact_path, True
        if candidate_mode == "heightmap":
            artifact_id, artifact_path = _thumbnail_artifact_candidate(detail, "normalized_heightmap", "normalized_heightmap_display", "raw_heightmap_preview", "heightmap")
            if artifact_path:
                path = safe_take_file(settings, detail.take_id, artifact_path)
                if path is not None and path.is_file():
                    source = "normalized_heightmap" if artifact_id and "normalized_heightmap" in artifact_id else "heightmap"
                    return path, source, artifact_id, artifact_path, source == "normalized_heightmap"
            groups = detail.assets or {}
            payload = groups.get("heightmap") or {}
            for key in ("heightmap", "height"):
                filename = payload.get(key)
                if not filename:
                    continue
                path = settings.incoming_dir / detail.take_id / str(filename)
                if path.is_file():
                    return path, "heightmap", key, str(filename), False
            return None, "heightmap", artifact_id, artifact_path, False
        if candidate_mode == "reflectance":
            groups = detail.assets or {}
            for group, keys in (("reflectance", ["reflectance"]), ("rgb", ["rgb"]), ("laser_rgb", ["laser_rgb"])):
                payload = groups.get(group) or {}
                for key in keys:
                    filename = payload.get(key)
                    if not filename:
                        continue
                    path = settings.incoming_dir / detail.take_id / str(filename)
                    if path.is_file():
                        return path, "reflectance", key, str(filename), group == "rgb"
            source_path_auto, resolved_name = resolve_source_image_path(settings, detail)
            if source_path_auto is not None and source_path_auto.is_file():
                return source_path_auto, "source_crop", None, resolved_name, True
            return None, "reflectance", None, None, True
        return None, "placeholder", None, None, False

    if requested_mode == "auto":
        preferred = ["heightmap", "classification_overlay", "reflectance"] if semantic in {"morphology", "geometry", "height"} else ["reflectance", "classification_overlay", "heightmap"]
    else:
        preferred = [requested_mode]
        if requested_mode == "heightmap":
            preferred.extend(["classification_overlay", "reflectance"])
        elif requested_mode == "classification_overlay":
            preferred.extend(["heightmap", "reflectance"])
        elif requested_mode == "reflectance":
            preferred.extend(["classification_overlay", "heightmap"])

    for index, candidate_mode in enumerate(preferred):
        path, source, artifact_id, artifact_path, candidate_colorized = resolve_candidate(candidate_mode)
        if path is None:
            continue
        source_path = path
        resolved_source = source
        resolved_artifact_id = artifact_id
        resolved_artifact_path = artifact_path
        colorized = candidate_colorized
        if index > 0:
            fallback_used = True
            fallback_reason = f"{preferred[0]} unavailable"
        break

    if source_path is None:
        summary = get_take_summary(settings, detail.take_id)
        thumb = summary.thumbnail_path if summary is not None else None
        take_thumb = safe_take_file(settings, detail.take_id, thumb) if isinstance(thumb, str) and thumb else None
        if take_thumb is not None and take_thumb.is_file():
            source_path = take_thumb
            resolved_source = "take_thumbnail"
            resolved_artifact_path = thumb
            colorized = True
            fallback_used = True
            fallback_reason = fallback_reason or f"{preferred[0]} unavailable"
        else:
            fallback_used = True
            fallback_reason = fallback_reason or f"{preferred[0]} unavailable"

    return {
        "requested_mode": requested_mode,
        "resolved_source": resolved_source,
        "resolved_artifact_id": resolved_artifact_id,
        "resolved_artifact_path": resolved_artifact_path,
        "colorized": colorized,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "source_path": source_path,
        "bbox": bbox,
    }


@app.get("/api/takes/{take_id}/objects/{object_id}/thumbnail-info")
def take_object_thumbnail_info(
    take_id: str,
    object_id: str,
    mode: str = "auto",
    semantic_group: str | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> dict[str, Any]:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")
    plan = _build_thumbnail_plan(settings, detail, object_id, mode=mode, semantic_group=semantic_group)
    return {
        "requested_mode": plan["requested_mode"],
        "resolved_source": plan["resolved_source"],
        "resolved_artifact_id": plan["resolved_artifact_id"],
        "resolved_artifact_path": plan["resolved_artifact_path"],
        "colorized": plan["colorized"],
        "fallback_used": plan["fallback_used"],
        "fallback_reason": plan["fallback_reason"],
    }


@app.get("/api/takes/{take_id}/objects/{object_id}/thumbnail")
def take_object_thumbnail(
    take_id: str,
    object_id: str,
    size: int = 56,
    pad: float = 0.18,
    mode: str = "auto",
    semantic_group: str | None = None,
    settings: ApiSettings = Depends(get_settings),
) -> FileResponse:
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")

    size_px = max(40, min(int(size), 256))
    pad_ratio = max(0.0, min(float(pad), 0.8))
    cache_dir = settings.incoming_dir / take_id / ".stage_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_thumbnail_plan(settings, detail, object_id, mode=mode, semantic_group=semantic_group)
    source_path: Path | None = plan.get("source_path")
    bbox = plan.get("bbox")
    resolved_source = str(plan.get("resolved_source") or "placeholder")
    resolved_artifact_id = str(plan.get("resolved_artifact_id") or "").strip() or "none"
    resolved_artifact_path = str(plan.get("resolved_artifact_path") or "").strip() or "none"
    cache_name = f"obj_thumb_{str(object_id).strip()}_{size_px}_{int(pad_ratio * 100)}_{mode}_{resolved_source}_{resolved_artifact_id}_{abs(hash(resolved_artifact_path)) % 1000000}.jpg"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        return FileResponse(cache_path)

    if source_path is None or not source_path.is_file():
        return FileResponse(_neutral_placeholder(cache_path, size_px))

    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            if bbox is not None:
                x, y, w, h = bbox
                cx = x + (w / 2.0)
                cy = y + (h / 2.0)
                side = max(8.0, max(w, h) * (1.0 + 2.0 * pad_ratio))
                left = max(0, int(round(cx - side / 2.0)))
                top = max(0, int(round(cy - side / 2.0)))
                right = min(image.width, int(round(cx + side / 2.0)))
                bottom = min(image.height, int(round(cy + side / 2.0)))
                if right > left and bottom > top:
                    image = image.crop((left, top, right, bottom))
            thumb = ImageOps.fit(image, (size_px, size_px), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
            thumb.save(cache_path, format="JPEG", quality=82, optimize=True)
    except Exception:
        return FileResponse(_neutral_placeholder(cache_path, size_px))

    return FileResponse(cache_path)


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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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
