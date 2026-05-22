from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FinalObject(BaseModel):
    id: str
    acquisition_group_id: str
    candidate_2d_id: str | None = None
    candidate_25d_id: str | None = None
    bbox_px: list[float] | None = None
    centroid_px: list[float] | None = None
    centroid_world: list[float] | None = None
    measurements: dict[str, Any] = Field(default_factory=dict)
    appearance: dict[str, Any] = Field(default_factory=dict)
    geometry: dict[str, Any] = Field(default_factory=dict)
    final_class: str = "Unknown"
    final_class_group: str = "Unknown"
    confidence: float | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class FusionResult(BaseModel):
    id: str
    acquisition_group_id: str
    fusion_run_id: str
    matched_objects: list[dict[str, Any]] = Field(default_factory=list)
    unmatched_2d_candidates: list[dict[str, Any]] = Field(default_factory=list)
    unmatched_25d_candidates: list[dict[str, Any]] = Field(default_factory=list)
    final_objects: list[FinalObject] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
