from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TakeStatus = Literal["incoming", "processed", "failed"]
Decision = Literal["accept", "reject", "review"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    data_dir: str
    incoming_count: int
    processed_count: int


class RuntimeState(BaseModel):
    status: str = "idle"
    latest_take_id: str | None = None
    latest_processed_take_id: str | None = None
    message: str | None = None
    updated_at: str | None = None
    acquisition_connected: bool = False
    latest_frame_timestamp: str | None = None
    processing_lag_ms: float | None = None
    queue_size: int = 0
    dropped_frames: int = 0
    last_successful_processing: str | None = None
    current_session: str | None = None
    active_calibration: str | None = None
    acquisition_source: str | None = None
    acquisition_source_details: dict[str, Any] = Field(default_factory=dict)
    preview_available: bool = False
    preview_timestamp: str | None = None
    preview_fps_estimate: float | None = None
    preview_stale: bool = True
    preview_source: str | None = None
    preview_path: str | None = None
    stale: bool = True
    throughput: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TakeSummary(BaseModel):
    take_id: str
    status: TakeStatus
    has_ready: bool
    has_done: bool
    created_at: str | None = None
    decision: Decision | None = None
    engine: str | None = None
    object_count: int | None = None
    ball_count: int | None = None
    plane_found: bool | None = None
    processing_time_ms: float | None = None
    warning_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    calibration_id: str | None = None
    calibration_file: str | None = None
    calibration_resolution_source: str | None = None
    modalities: list[str] = Field(default_factory=list)
    assets: dict[str, dict[str, str]] = Field(default_factory=dict)
    frame_count: int | None = None
    frameset: dict[str, Any] | None = None
    session_id: str | None = None
    frameset_id: str | None = None
    throughput: dict[str, Any] | None = None
    processing_by_family: list[dict[str, Any]] = Field(default_factory=list)
    friendly_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    validation_status: str | None = None
    expected_class: str | None = None
    expected_diameter_mm: float | None = None
    thumbnail_path: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    experiment_session_id: str | None = None
    experiment_session_name: str | None = None
    latest_run_status: str | None = None


class TakeDetail(BaseModel):
    take_id: str
    status: TakeStatus
    has_ready: bool
    has_done: bool
    created_at: str | None = None
    metadata: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    modalities: list[str] = Field(default_factory=list)
    assets: dict[str, dict[str, str]] = Field(default_factory=dict)
    frame_count: int | None = None
    frameset: dict[str, Any] | None = None
    session_id: str | None = None
    frameset_id: str | None = None
    throughput: dict[str, Any] | None = None
    processing_by_family: list[dict[str, Any]] = Field(default_factory=list)
    take_metadata: dict[str, Any] = Field(default_factory=dict)
    run_history: list[dict[str, Any]] = Field(default_factory=list)
    object_annotations: list[dict[str, Any]] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    session_count: int = 0
    take_count: int = 0


class DatasetSessionSummary(BaseModel):
    id: str
    dataset_id: str
    name: str
    description: str | None = None
    calibration_id: str | None = None
    sensor_metadata: dict[str, Any] = Field(default_factory=dict)
    conveyor_metadata: dict[str, Any] = Field(default_factory=dict)
    lighting_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    take_count: int = 0
