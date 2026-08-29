from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass
class ProcessingUnitRuntimeTraceEntry:
    unit_id: str
    stage_id: str
    parent_id: str | None = None
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    parameters_used: dict[str, Any] = field(default_factory=dict)
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    trace_source: str = "runtime_unit_callbacks"
    trace_precision: str = "unit_level"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class ProcessingUnitTraceScope(AbstractContextManager["ProcessingUnitTraceScope"]):
    def __init__(self, recorder: "ProcessingUnitTraceRecorder", entry: ProcessingUnitRuntimeTraceEntry) -> None:
        self._recorder = recorder
        self.entry = entry
        self._skipped = False

    def __enter__(self) -> "ProcessingUnitTraceScope":
        self.entry.started_at = self.entry.started_at or _now_iso()
        self.entry.status = "running"
        self._recorder._store(self.entry)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.entry.completed_at = _now_iso()
            if self.entry.started_at and self.entry.completed_at:
                try:
                    start = datetime.fromisoformat(self.entry.started_at)
                    end = datetime.fromisoformat(self.entry.completed_at)
                    self.entry.duration_ms = max(0, int(round((end - start).total_seconds() * 1000.0)))
                except Exception:
                    self.entry.duration_ms = None
            if exc is not None:
                self.entry.status = "failed"
                self.error(f"{exc.__class__.__name__}: {exc}")
            elif self._skipped or self.entry.status == "skipped":
                self.entry.status = "skipped"
            elif self.entry.warnings and self.entry.status in {"pending", "running", "completed", "warning"}:
                self.entry.status = "warning"
            elif self.entry.status in {"pending", "running"}:
                self.entry.status = "completed"
        finally:
            self._recorder._store(self.entry)
        return False

    def set_status(self, status: str) -> None:
        try:
            self.entry.status = str(status)
        except Exception:
            return

    def skip(self, reason: str | None = None) -> None:
        self._skipped = True
        self.entry.status = "skipped"
        if reason:
            self.add_diagnostic("skip_reason", reason)

    def add_parameter(self, key: str, value: Any) -> None:
        try:
            self.entry.parameters_used[str(key)] = _json_safe(value)
        except Exception:
            return

    def add_parameters(self, values: dict[str, Any] | None) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            self.add_parameter(str(key), value)

    def add_input_artifact(self, artifact_id: str) -> None:
        try:
            artifact = str(artifact_id)
            if artifact and artifact not in self.entry.input_artifacts:
                self.entry.input_artifacts.append(artifact)
        except Exception:
            return

    def add_output_artifact(self, artifact_id: str) -> None:
        try:
            artifact = str(artifact_id)
            if artifact and artifact not in self.entry.output_artifacts:
                self.entry.output_artifacts.append(artifact)
        except Exception:
            return

    def add_metric(self, key: str, value: Any) -> None:
        try:
            self.entry.metrics[str(key)] = _json_safe(value)
        except Exception:
            return

    def add_diagnostic(self, key: str, value: Any) -> None:
        try:
            self.entry.diagnostics[str(key)] = _json_safe(value)
        except Exception:
            return

    def warn(self, message: str) -> None:
        try:
            text = str(message)
            if text:
                self.entry.warnings.append(text)
        except Exception:
            return

    def error(self, message: str) -> None:
        try:
            text = str(message)
            if text:
                self.entry.errors.append(text)
        except Exception:
            return


@dataclass
class ProcessingUnitTraceRecorder:
    entries: dict[str, ProcessingUnitRuntimeTraceEntry] = field(default_factory=dict)

    def _store(self, entry: ProcessingUnitRuntimeTraceEntry) -> None:
        self.entries[entry.unit_id] = entry

    def trace_unit(
        self,
        unit_id: str,
        *,
        stage_id: str,
        parent_id: str | None = None,
    ) -> ProcessingUnitTraceScope:
        entry = self.entries.get(unit_id) or ProcessingUnitRuntimeTraceEntry(
            unit_id=str(unit_id),
            stage_id=str(stage_id),
            parent_id=str(parent_id) if parent_id else None,
        )
        return ProcessingUnitTraceScope(self, entry)

    def to_payload(self) -> dict[str, Any]:
        unit_results = {
            unit_id: entry.to_dict()
            for unit_id, entry in sorted(self.entries.items())
        }
        return {
            "trace_source": "runtime_unit_callbacks",
            "trace_precision": "unit_level",
            "unit_results": unit_results,
            "units": unit_results,
        }
