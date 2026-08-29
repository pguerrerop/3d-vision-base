from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from vision_3d_acquisition.contracts.modalities import CaptureModality, format_modalities
from vision_3d_acquisition.pipelines.runtime_trace import ProcessingUnitTraceRecorder, ProcessingUnitTraceScope
from vision_3d_acquisition.utils.profiling import ProcessingProfiler


@dataclass
class PipelineContext:
    """Execution context shared by pipeline stages."""

    profiler: ProcessingProfiler = field(default_factory=ProcessingProfiler)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    debug_outputs: list[dict[str, Any]] = field(default_factory=list)
    processing_artifacts: list[dict[str, Any]] = field(default_factory=list)
    runtime_trace: ProcessingUnitTraceRecorder = field(default_factory=ProcessingUnitTraceRecorder)

    def set_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    def get_artifact(self, name: str, default: Any = None) -> Any:
        return self.artifacts.get(name, default)

    def require_artifact(self, name: str) -> Any:
        if name not in self.artifacts:
            raise KeyError(f"Missing required artifact: {name}")
        return self.artifacts[name]

    def add_debug_output(self, kind: str, path: Path | str, **metadata: Any) -> None:
        payload: dict[str, Any] = {"kind": kind, "path": str(path)}
        payload.update(metadata)
        self.debug_outputs.append(payload)

    def trace_unit(
        self,
        unit_id: str,
        *,
        stage_id: str,
        parent_id: str | None = None,
    ) -> ProcessingUnitTraceScope:
        return self.runtime_trace.trace_unit(unit_id, stage_id=stage_id, parent_id=parent_id)

    def add_processing_artifact(
        self,
        *,
        artifact_id: str,
        stage_id: str,
        kind: str,
        title: str,
        object_id: int | None = None,
        description: str | None = None,
        path: str | None = None,
        mime_type: str | None = None,
        preview_available: bool = False,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        generated: str = "explicit",
        overlay_type: str | None = None,
        coordinate_space: str | None = None,
        target_artifact_id: str | None = None,
        geometry: dict[str, Any] | None = None,
        style: dict[str, Any] | None = None,
        source_artifact_ids: list[str] | None = None,
        approximate: bool = False,
        overlay_warnings: list[str] | None = None,
        projection_type: str | None = None,
        projection_coordinate_system: dict[str, Any] | None = None,
        projection_metadata: dict[str, Any] | None = None,
        projection_transform_id: str | None = None,
    ) -> None:
        self.processing_artifacts.append(
            {
                "artifact_id": artifact_id,
                "stage_id": stage_id,
                "object_id": object_id,
                "kind": kind,
                "title": title,
                "description": description,
                "path": path,
                "mime_type": mime_type,
                "preview_available": preview_available,
                "metadata": dict(metadata or {}),
                "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
                "generated": generated,
                "overlay_type": overlay_type,
                "coordinate_space": coordinate_space,
                "target_artifact_id": target_artifact_id,
                "geometry": dict(geometry or {}),
                "style": dict(style or {}),
                "source_artifact_ids": list(source_artifact_ids or []),
                "approximate": approximate,
                "overlay_warnings": list(overlay_warnings or []),
                "projection_type": projection_type,
                "projection_coordinate_system": dict(projection_coordinate_system or {}),
                "projection_metadata": dict(projection_metadata or {}),
                "projection_transform_id": projection_transform_id,
            }
        )


class PipelineStage(Protocol):
    """Contract for a single processing stage."""

    name: str
    category: str
    required_modalities: tuple[CaptureModality, ...]
    produced_modalities: tuple[CaptureModality, ...]
    produced_artifacts: tuple[str, ...]
    stage_category: str | None

    def run(self, context: PipelineContext) -> None:
        ...


@dataclass
class PipelineResult:
    success: bool
    artifacts: dict[str, Any]
    metrics: dict[str, Any]
    profiling: dict[str, Any]
    debug_outputs: list[dict[str, Any]]
    processing_artifacts: list[dict[str, Any]]


class PipelineRunner:
    """Execute stages in order with shared context."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self.stages = stages

    def run(self, context: PipelineContext | None = None) -> PipelineResult:
        active_context = context or PipelineContext()
        try:
            for stage in self.stages:
                with active_context.profiler.stage(stage.name, category=stage.category) as profile_stage:
                    self._validate_modalities(stage, active_context, profile_stage.metadata)
                    stage.run(active_context)
            active_context.profiler.finish()
        except Exception:
            active_context.profiler.finish()
            raise
        return PipelineResult(
            success=True,
            artifacts=dict(active_context.artifacts),
            metrics=dict(active_context.metrics),
            profiling=active_context.profiler.as_dict(),
            debug_outputs=list(active_context.debug_outputs),
            processing_artifacts=list(active_context.processing_artifacts),
        )

    @staticmethod
    def _validate_modalities(stage: PipelineStage, context: PipelineContext, profile_metadata: dict[str, Any]) -> None:
        required = tuple(getattr(stage, "required_modalities", ()) or ())
        if not required:
            return
        available = context.get_artifact("modalities")
        if available is None:
            reference = context.get_artifact("capture_ref")
            available = getattr(reference, "modalities", None)
        if available is None:
            return
        available_set = set(available)
        missing = [modality for modality in required if modality not in available_set]
        if not missing:
            return
        message = (
            f"Stage {stage.name} requires {format_modalities(missing)} "
            f"but take has {format_modalities(available)}"
        )
        context.metrics.setdefault("modality_errors", []).append(
            {
                "stage": stage.name,
                "required_modalities": list(required),
                "available_modalities": list(available),
                "missing_modalities": missing,
                "message": message,
            }
        )
        profile_metadata["modality_error"] = message
        profile_metadata["required_modalities"] = list(required)
        profile_metadata["available_modalities"] = list(available)
        raise ValueError(message)
