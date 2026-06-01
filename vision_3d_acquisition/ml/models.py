from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


FEATURE_GROUPS = (
    "geometry_2d",
    "ellipse_fit",
    "morphology",
    "height_25d",
    "deformation",
    "reflectance",
    "rgb_texture",
    "fusion_basic",
)

SPLIT_STRATEGIES = ("random", "stratified", "by-session", "by-dataset", "chronological")
CLASSIFIER_TYPES = ("heuristic", "logistic_regression", "random_forest", "gradient_boosting", "svm", "pytorch", "onnx", "tensorflow")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MLDataset(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str = Field(default_factory=now_iso)
    source_dataset_ids: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    label_schema: dict[str, Any] = Field(default_factory=dict)
    object_count: int = 0
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class FeatureSetDefinition(BaseModel):
    id: str
    name: str
    modality_support: list[str] = Field(default_factory=lambda: ["heightmap", "rgb", "point_cloud"])
    feature_groups: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    description: str | None = None


class MLExperiment(BaseModel):
    id: str
    name: str
    classifier_type: str
    classifier_types: list[str] = Field(default_factory=list)
    experiment_mode: str = "single"
    feature_set_id: str
    dataset_id: str
    split_strategy: str
    training_config: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class TrainedModel(BaseModel):
    id: str
    name: str
    version: str = "1.0.0"
    classifier_family: str
    training_experiment_id: str
    feature_schema: dict[str, Any] = Field(default_factory=dict)
    label_schema: dict[str, Any] = Field(default_factory=dict)
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    compatible_pipeline_families: list[str] = Field(default_factory=lambda: ["25d"])
    promoted: bool = False
    deployment_notes: str | None = None


class ModelDeployment(BaseModel):
    id: str
    model_id: str
    target_pipeline_id: str
    target_stage_id: str
    pipeline_family: str = "25d"
    deployed_at: str = Field(default_factory=now_iso)
    deployed_by: str = "system"
    deployment_mode: str = "explicit"
    rollback_model_id: str | None = None
    notes: str | None = None
    status: Literal["inactive", "active", "superseded", "rolled_back", "invalid"] = "inactive"
    activated_at: str | None = None
    deactivated_at: str | None = None
    superseded_by: str | None = None
    rollback_from: str | None = None
    runtime_validation: dict[str, Any] = Field(default_factory=dict)
    compatibility_snapshot: dict[str, Any] = Field(default_factory=dict)
