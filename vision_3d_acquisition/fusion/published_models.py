from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PublishedInspectionStatus = Literal["pending", "complete", "incomplete", "failed"]
PublishedOverallDecision = Literal["accept", "reject", "review", "unknown"]


class PublishedInspectionObject(BaseModel):
    final_object_id: str
    final_class: str
    final_class_group: str
    confidence: float | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    key_measurements: dict[str, Any] = Field(default_factory=dict)
    display_label: str
    sort_order: int


class PublishedInspectionResult(BaseModel):
    published_result_id: str
    acquisition_group_id: str
    fusion_run_id: str
    station_id: str | None = None
    session_id: str | None = None
    timestamp: str
    status: PublishedInspectionStatus
    overall_decision: PublishedOverallDecision
    primary_class: str | None = None
    primary_class_group: str | None = None
    confidence: float | None = None
    class_counts: dict[str, int] = Field(default_factory=dict)
    objects: list[PublishedInspectionObject] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    display_artifacts: dict[str, str | None] = Field(default_factory=dict)
    source_refs: dict[str, Any] = Field(default_factory=dict)
