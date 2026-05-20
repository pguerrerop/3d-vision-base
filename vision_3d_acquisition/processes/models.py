from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ProcessCategory = Literal["image_2d", "image_3d", "hybrid"]
InputType = Literal["grayscale_image", "reflectance_image", "trispector_reflectance_png", "point_cloud", "rgb_image"]
StepStatus = Literal["idle", "running", "success", "warning", "failed", "skipped"]
PipelineStatus = Literal["idle", "ready", "running", "success", "warning", "failed"]


class PipelineStepConfig(BaseModel):
    step_id: str
    algorithm_key: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    ui_state: dict[str, Any] = Field(default_factory=dict)


class StepDefinition(BaseModel):
    step_id: str
    title: str
    namespace: str
    default_algorithm_key: str
    description: str = ""
    optional: bool = False
    input_artifact_types: list[str] = Field(default_factory=list)
    output_artifact_types: list[str] = Field(default_factory=list)
    overlay_types: list[str] = Field(default_factory=list)
    params_schema: dict[str, Any] = Field(default_factory=dict)


class ProcessTemplate(BaseModel):
    id: str
    name: str
    category: ProcessCategory
    description: str
    supported_input_types: list[InputType] = Field(default_factory=list)
    steps: list[StepDefinition] = Field(default_factory=list)
    default_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ui_metadata: dict[str, Any] = Field(default_factory=dict)


class StepRunReport(BaseModel):
    step_id: str
    algorithm_key: str
    status: StepStatus
    duration_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    overlay_ids: list[str] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class PipelineRunRecord(BaseModel):
    run_id: str
    timestamp: datetime
    status: PipelineStatus
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    overlays: list[dict[str, Any]] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    step_reports: list[StepRunReport] = Field(default_factory=list)


class PipelineRuntimeState(BaseModel):
    status: PipelineStatus = "idle"
    last_run_id: str | None = None
    last_error: str | None = None


class PipelineInstance(BaseModel):
    id: str
    name: str
    template_id: str
    pipeline_family: Literal["2d", "3d", "generic"] = "2d"
    supported_input_type: InputType
    configured_steps: list[PipelineStepConfig] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    runtime_state: PipelineRuntimeState = Field(default_factory=PipelineRuntimeState)
    execution_history: list[PipelineRunRecord] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class RecipeVersion(BaseModel):
    id: str
    pipeline_instance_id: str
    source_run_id: str | None = None
    type: Literal["poc_baseline", "production", "regression"] = "poc_baseline"
    notes: str | None = None
    created_at: datetime
    template_id: str
    pipeline_snapshot: dict[str, Any] = Field(default_factory=dict)


class StepExecutionContext(BaseModel):
    pipeline_id: str
    run_id: str
    step: PipelineStepConfig
    available_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    overlays: list[dict[str, Any]] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
