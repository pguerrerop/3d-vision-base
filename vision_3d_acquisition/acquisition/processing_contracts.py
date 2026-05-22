from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

AcquisitionProcessingStatus = Literal[
    "acquired",
    "ready_for_processing",
    "processing_binding_missing",
    "processing_enqueued",
    "processing_running",
    "processing_completed",
    "processing_failed",
    "ready_not_processed",
]


@dataclass
class AcquiredTakeReady:
    take_id: str
    source_id: str
    modality: str
    modality_family: str | None = None
    acquisition_process_id: str | None = None
    acquisition_run_id: str | None = None
    acquisition_group_id: str | None = None
    session_id: str | None = None
    asset_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AcquisitionProcessingDecision:
    ok: bool
    status: AcquisitionProcessingStatus
    normalized_modality: str
    original_modality: str
    warning: str | None = None
    warnings: list[str] = field(default_factory=list)
    pipeline_id: str | None = None
    recipe_version_id: str | None = None
    calibration_profile_id: str | None = None
    run_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
