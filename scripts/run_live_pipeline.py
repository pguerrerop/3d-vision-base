#!/usr/bin/env python3
"""Watch incoming takes and process them in a single local loop."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.ball_inspection import run_ball_inspection_flow  # noqa: E402
from vision_3d_acquisition.processing.pipeline import load_processing_config, process_take  # noqa: E402
from vision_3d_acquisition.runtime_config import resolve_calibration  # noqa: E402
from vision_3d_acquisition.state.runtime_status import update_runtime_status  # noqa: E402
from vision_3d_acquisition.state.sessions import attach_take_to_session, ensure_session, generate_session_id  # noqa: E402
from vision_3d_acquisition.vision_core.serialization import write_processing_outputs  # noqa: E402


_STOP = False


def _handle_signal(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


@dataclass
class RollingMetrics:
    acquisition_times: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    processing_times: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    queue_wait_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    write_overhead_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    debug_overhead_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    dropped_frames: int = 0

    def record_acquisition(self, timestamp_s: float) -> None:
        self.acquisition_times.append(timestamp_s)

    def record_processing(self, started_s: float, ended_s: float, *, queue_wait_ms: float, write_overhead_ms: float, debug_overhead_ms: float) -> None:
        self.processing_times.append(ended_s)
        self.latencies_ms.append(max(0.0, (ended_s - started_s) * 1000.0))
        self.queue_wait_ms.append(max(0.0, queue_wait_ms))
        self.write_overhead_ms.append(max(0.0, write_overhead_ms))
        self.debug_overhead_ms.append(max(0.0, debug_overhead_ms))

    def fps(self, samples: deque[float]) -> float | None:
        if len(samples) < 2:
            return None
        elapsed = samples[-1] - samples[0]
        if elapsed <= 0:
            return None
        return (len(samples) - 1) / elapsed

    def average(self, samples: deque[float]) -> float | None:
        if not samples:
            return None
        return float(sum(samples) / len(samples))

    def maximum(self, samples: deque[float]) -> float | None:
        if not samples:
            return None
        return float(max(samples))

    def payload(self, queue_size: int) -> dict[str, Any]:
        acquisition_fps = self.fps(self.acquisition_times)
        processing_fps = self.fps(self.processing_times)
        average_latency = self.average(self.latencies_ms)
        average_queue_wait = self.average(self.queue_wait_ms)
        warnings: list[str] = []
        if acquisition_fps is not None and processing_fps is not None and processing_fps < acquisition_fps:
            warnings.append("processing slower than acquisition")
        if queue_size > 1:
            warnings.append("queue buildup")
        if (self.average(self.write_overhead_ms) or 0.0) > 200.0:
            warnings.append("excessive export time")
        if (self.average(self.debug_overhead_ms) or 0.0) > 250.0:
            warnings.append("debug rendering bottleneck")
        return {
            "acquisition_fps": acquisition_fps,
            "processing_fps": processing_fps,
            "average_processing_latency_ms": average_latency,
            "max_processing_latency_ms": self.maximum(self.latencies_ms),
            "average_queue_wait_ms": average_queue_wait,
            "render_debug_overhead_ms": self.average(self.debug_overhead_ms),
            "export_write_overhead_ms": self.average(self.write_overhead_ms),
            "warnings": warnings,
        }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def list_ready_unprocessed(data_dir: Path) -> list[Path]:
    incoming = data_dir / "incoming"
    processed = data_dir / "processed"
    if not incoming.exists():
        return []
    candidates: list[tuple[str, Path]] = []
    for take_dir in incoming.iterdir():
        if not take_dir.is_dir() or not (take_dir / "READY").is_file():
            continue
        if (processed / take_dir.name / "DONE").is_file():
            continue
        metadata = read_json(take_dir / "metadata.json") if (take_dir / "metadata.json").is_file() else {}
        created = str(metadata.get("created_at") or take_dir.name)
        candidates.append((created, take_dir))
    candidates.sort(key=lambda item: item[0])
    return [item[1] for item in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-process live processing loop for incoming takes.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data root")
    parser.add_argument("--engine", choices=("legacy", "native"), default="legacy")
    parser.add_argument("--config", default=Path("config/processing.yaml"), type=Path)
    parser.add_argument("--calibration", default=None, type=Path)
    parser.add_argument("--session", default=None, help="Optional existing session id override")
    parser.add_argument("--poll-interval", default=0.75, type=float)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--demo-mode", action="store_true")
    parser.add_argument("--skip-debug-images", action="store_true")
    parser.add_argument("--prefer-fast-cloud", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    for child in ("incoming", "processed", "state", "events", "sessions"):
        (data_dir / child).mkdir(parents=True, exist_ok=True)
    active_session = args.session or generate_session_id()
    ensure_session(
        data_dir / "sessions",
        active_session,
        acquisition_mode="live",
        notes="demo mode live loop" if args.demo_mode else "live processing loop",
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    metrics = RollingMetrics()
    calibration_resolution = resolve_calibration(args.calibration)
    config = load_processing_config(args.config)
    update_runtime_status(
        data_dir / "state",
        status="running",
        acquisition_connected=True,
        current_session=active_session,
        active_calibration=str(calibration_resolution.calibration) if calibration_resolution.calibration else None,
        acquisition_source="live_loop",
        message="Live processing loop started.",
    )

    while not _STOP:
        queue = list_ready_unprocessed(data_dir)
        queue_size = len(queue)
        now_s = time.time()
        if queue_size == 0:
            update_runtime_status(
                data_dir / "state",
                status="running",
                acquisition_connected=True,
                queue_size=0,
                throughput=metrics.payload(0),
                warnings=[],
                message="Live loop idle; waiting for incoming takes.",
            )
            time.sleep(max(0.1, args.poll_interval))
            continue
        take_dir = queue[0]
        metadata_path = take_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        metrics.record_acquisition(now_s)
        take_id = take_dir.name
        created_at = metadata.get("created_at")
        if not metadata.get("session_id"):
            metadata["session_id"] = active_session
            write_json(metadata_path, metadata)
        attach_take_to_session(data_dir / "sessions", metadata["session_id"], take_id, take_metadata=metadata)
        wait_ms = max(0.0, (now_s - _parse_time(created_at, fallback=now_s)) * 1000.0)
        update_runtime_status(
            data_dir / "state",
            status="processing",
            latest_take_id=take_id,
            latest_frame_timestamp=created_at,
            processing_lag_ms=wait_ms,
            queue_size=queue_size,
            current_session=metadata["session_id"],
            acquisition_source=str(metadata.get("source") or "live_loop"),
            warnings=metrics.payload(queue_size)["warnings"],
            message=f"Processing {take_id}",
        )
        started = time.perf_counter()
        output_dir = data_dir / "processed" / take_id
        try:
            if args.engine == "legacy":
                result = process_take(
                    take_dir,
                    output_dir,
                    config=config,
                    calibration=calibration_resolution.calibration,
                    calibration_resolution_source=calibration_resolution.source,
                    skip_debug_images=args.skip_debug_images,
                    prefer_fast_cloud=args.prefer_fast_cloud,
                )
                payload = result.model_dump(mode="json")
                profiling_payload = payload.get("profiling") if isinstance(payload.get("profiling"), dict) else {}
                debug_overhead = float(profiling_payload.get("debug_artifacts_ms") or 0.0)
                write_started = time.perf_counter()
                throughput = metrics.payload(queue_size)
                payload["throughput"] = throughput
                payload["runtime_acquisition"] = {
                    "acquisition_connected": True,
                    "acquisition_source": str(metadata.get("source") or "live_loop"),
                    "latest_frame_timestamp": created_at,
                    "queue_size": queue_size,
                    "queue_wait_ms": wait_ms,
                    "processing_lag_ms": wait_ms,
                    "dropped_frames": metrics.dropped_frames,
                }
                write_processing_outputs(
                    data_dir,
                    output_dir,
                    payload,
                    runtime_message="Live processing loop active.",
                    runtime_metrics={
                        "queue_size": max(0, queue_size - 1),
                        "processing_lag_ms": wait_ms,
                        "current_session": metadata.get("session_id"),
                        "acquisition_source": str(metadata.get("source") or "live_loop"),
                        "latest_frame_timestamp": created_at,
                        "throughput": throughput,
                        "warnings": throughput.get("warnings", []),
                    },
                )
                write_overhead_ms = (time.perf_counter() - write_started) * 1000.0
            else:
                native = run_ball_inspection_flow(
                    data_dir,
                    take_id=take_id,
                    config_path=args.config,
                    calibration=calibration_resolution.calibration,
                    calibration_resolution_source=calibration_resolution.source,
                    skip_debug_images=args.skip_debug_images,
                    prefer_fast_cloud=args.prefer_fast_cloud,
                )
                payload = native.result_payload
                payload["session_id"] = metadata.get("session_id")
                payload["throughput"] = metrics.payload(queue_size)
                write_json(native.output_dir / "result.json", payload)
                profiling_payload = payload.get("profiling") if isinstance(payload.get("profiling"), dict) else {}
                debug_overhead = float(profiling_payload.get("debug_artifacts_ms") or 0.0)
                write_overhead_ms = 0.0
            ended = time.perf_counter()
            metrics.record_processing(
                started_s=started,
                ended_s=ended,
                queue_wait_ms=wait_ms,
                write_overhead_ms=write_overhead_ms,
                debug_overhead_ms=debug_overhead,
            )
            throughput = metrics.payload(max(0, queue_size - 1))
            update_runtime_status(
                data_dir / "state",
                status="running",
                latest_processed_take_id=take_id,
                last_successful_processing=payload.get("processed_at"),
                queue_size=max(0, queue_size - 1),
                throughput=throughput,
                processing_lag_ms=wait_ms,
                warnings=throughput.get("warnings", []),
                stale=False,
                message=f"Processed {take_id}",
            )
            if args.profile:
                print(json.dumps({"take_id": take_id, "throughput": throughput}, indent=2))
        except Exception as exc:  # noqa: BLE001
            update_runtime_status(
                data_dir / "state",
                status="error",
                message=f"Failed processing {take_id}: {exc}",
                queue_size=queue_size,
            )
            print(f"error: {exc}", file=sys.stderr)
            time.sleep(max(0.1, args.poll_interval))
            continue
        time.sleep(max(0.05, args.poll_interval))

    update_runtime_status(
        data_dir / "state",
        status="stopped",
        acquisition_connected=False,
        stale=True,
        message="Live processing loop stopped gracefully.",
    )
    return 0


def _parse_time(value: Any, *, fallback: float) -> float:
    if not value:
        return fallback
    text = str(value).replace("Z", "+00:00")
    try:
        import datetime as _dt

        return _dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return fallback


if __name__ == "__main__":
    raise SystemExit(main())
