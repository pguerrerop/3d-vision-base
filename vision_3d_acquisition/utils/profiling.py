from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ProcessingStage:
    name: str
    category: str
    started_at: float
    ended_at: float | None = None
    duration_ms: float = 0.0
    input_points: int | None = None
    output_points: int | None = None
    input_objects: int | None = None
    output_objects: int | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, ended_at: float | None = None) -> None:
        self.ended_at = ended_at if ended_at is not None else time.perf_counter()
        self.duration_ms = max(0.0, (self.ended_at - self.started_at) * 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 3),
            "input_points": self.input_points,
            "output_points": self.output_points,
            "input_objects": self.input_objects,
            "output_objects": self.output_objects,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class ProcessingProfiler:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self._ended_at: float | None = None
        self.stages: list[ProcessingStage] = []

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str,
        input_points: int | None = None,
        output_points: int | None = None,
        input_objects: int | None = None,
        output_objects: int | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ProcessingStage]:
        stage = ProcessingStage(
            name=name,
            category=category,
            started_at=time.perf_counter(),
            input_points=input_points,
            output_points=output_points,
            input_objects=input_objects,
            output_objects=output_objects,
            notes=notes,
            metadata=dict(metadata or {}),
        )
        try:
            yield stage
        finally:
            stage.finish()
            self.stages.append(stage)

    def record_skipped(
        self,
        name: str,
        *,
        category: str,
        notes: str,
        input_points: int | None = None,
        input_objects: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProcessingStage:
        now = time.perf_counter()
        stage = ProcessingStage(
            name=name,
            category=category,
            started_at=now,
            ended_at=now,
            duration_ms=0.0,
            input_points=input_points,
            input_objects=input_objects,
            notes=notes,
            metadata=dict(metadata or {}),
        )
        self.stages.append(stage)
        return stage

    def finish(self) -> None:
        self._ended_at = time.perf_counter()

    @property
    def total_ms(self) -> float:
        ended_at = self._ended_at if self._ended_at is not None else time.perf_counter()
        return max(0.0, (ended_at - self.started_at) * 1000.0)

    def aggregate_ms(self, *categories: str) -> float:
        wanted = set(categories)
        return sum(stage.duration_ms for stage in self.stages if stage.category in wanted)

    def production_ms(self, *, include_output: bool = False) -> float:
        excluded = {"debug_artifacts"}
        if not include_output:
            excluded.add("output")
        return sum(stage.duration_ms for stage in self.stages if stage.category not in excluded)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 3),
            "production_ms": round(self.production_ms(), 3),
            "debug_artifacts_ms": round(self.aggregate_ms("debug_artifacts"), 3),
            "io_ms": round(self.aggregate_ms("io"), 3),
            "stages": [stage.as_dict() for stage in self.stages],
        }
