from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.state.runtime_status import update_runtime_status


def write_result_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_processing_outputs(
    data_dir: Path,
    output_dir: Path,
    result_payload: dict[str, Any],
    *,
    runtime_message: str,
    runtime_metrics: dict[str, Any] | None = None,
) -> None:
    """Write the shared processing completion artifacts consumed by UI/API clients."""
    write_result_json(output_dir / "result.json", result_payload)
    (output_dir / "DONE").touch()
    write_result_json(data_dir / "state" / "latest.json", _latest_state_payload(result_payload))
    runtime_update = {
        "status": "ready",
        "latest_take_id": result_payload["take_id"],
        "latest_processed_take_id": result_payload["take_id"],
        "message": runtime_message,
        "last_successful_processing": result_payload["processed_at"],
        "updated_at": result_payload["processed_at"],
        "current_session": result_payload.get("session_id"),
    }
    if runtime_metrics:
        runtime_update.update(runtime_metrics)
    update_runtime_status(data_dir / "state", **runtime_update)
    event_path = data_dir / "events" / "events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_processed_event_payload(result_payload)) + "\n")


def _latest_state_payload(result_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "take_id": result_payload["take_id"],
        "processed_at": result_payload["processed_at"],
        "processing_mode": result_payload["processing_mode"],
        "algorithm_stage": result_payload["algorithm_stage"],
        "status": result_payload["status"],
        "summary": result_payload["summary"],
        "input_stats": result_payload.get("input_stats"),
        "plane_model": result_payload.get("plane_model"),
        "files": result_payload["files"],
        "timing_ms": result_payload["timing_ms"],
        "profiling": result_payload.get("profiling"),
        "calibration_id": result_payload.get("calibration_id"),
        "session_id": result_payload.get("session_id"),
        "frameset_id": result_payload.get("frameset_id"),
        "runtime_acquisition": result_payload.get("runtime_acquisition"),
        "throughput": result_payload.get("throughput"),
        "synchronization": result_payload.get("synchronization"),
        "acquisition_timestamps": result_payload.get("acquisition_timestamps"),
    }


def _processed_event_payload(result_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "processed",
        "take_id": result_payload["take_id"],
        "created_at": result_payload["processed_at"],
        "processing_mode": result_payload["processing_mode"],
        "algorithm_stage": result_payload["algorithm_stage"],
        "summary": result_payload["summary"],
        "calibration_id": result_payload.get("calibration_id"),
        "session_id": result_payload.get("session_id"),
        "frameset_id": result_payload.get("frameset_id"),
    }
