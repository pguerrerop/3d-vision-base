#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing  # noqa: E402
from vision_3d_acquisition.api.settings import ApiSettings  # noqa: E402
from vision_3d_acquisition.operations.summary import build_operations_card, update_operations_index, write_json, write_operations_card  # noqa: E402
from vision_3d_acquisition.runtime.process_registry import ProcessHeartbeat, RuntimeProcessRegistry  # noqa: E402

OK_STATUSES = {"ok", "success", "warning", "completed"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dedicated 2.5D worker for mining_steel_ball_classification_25d takes.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pipeline-id", default="mining_steel_ball_classification_25d")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-per-iteration", type=int, default=4)
    return parser.parse_args()


def _settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def runtime_state_path(data_dir: Path, take_id: str) -> Path:
    return data_dir / "incoming" / take_id / "runtime_state.json"


def _mark_state(data_dir: Path, take_id: str, state: str, patch: dict[str, Any] | None = None) -> dict[str, Any]:
    path = runtime_state_path(data_dir, take_id)
    current = read_json(path) or {
        "take_id": take_id,
        "state": "acquired",
        "source": "trispector_ftp",
        "created_at": now_iso(),
    }
    current["state"] = state
    current["updated_at"] = now_iso()
    if patch:
        current.update(patch)
    write_json(path, current)
    return current


def _take_metadata(data_dir: Path, take_id: str) -> dict[str, Any] | None:
    return read_json(data_dir / "incoming" / take_id / "metadata.json")


def _result_payload(data_dir: Path, take_id: str) -> dict[str, Any] | None:
    return read_json(data_dir / "processed" / take_id / "result.json")


def _is_25d_ready_take(data_dir: Path, take_id: str) -> bool:
    metadata = _take_metadata(data_dir, take_id)
    if not isinstance(metadata, dict):
        return False
    modalities = [str(item) for item in (metadata.get("modalities") or [])]
    return "heightmap" in modalities and (data_dir / "incoming" / take_id / "READY").is_file()


def _list_candidate_take_ids(data_dir: Path, retry_failed: bool) -> list[str]:
    incoming = data_dir / "incoming"
    if not incoming.is_dir():
        return []
    rows: list[tuple[float, str]] = []
    for take_dir in incoming.iterdir():
        if not take_dir.is_dir() or take_dir.name.startswith("."):
            continue
        take_id = take_dir.name
        if not _is_25d_ready_take(data_dir, take_id):
            continue
        state = read_json(runtime_state_path(data_dir, take_id)) or {}
        state_value = str(state.get("state") or "acquired")
        if state_value in {"processing", "completed"}:
            continue
        if state_value == "failed" and not retry_failed:
            continue
        if state_value not in {"acquired", "queued", "failed"}:
            continue
        created = (read_json(take_dir / "metadata.json") or {}).get("created_at")
        ts = 0.0
        try:
            if isinstance(created, str):
                ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = take_dir.stat().st_mtime
        rows.append((ts, take_id))
    rows.sort(key=lambda item: item[0])
    return [take_id for _, take_id in rows]


def _publish_card_and_index(data_dir: Path, take_id: str) -> dict[str, Any]:
    incoming_metadata = _take_metadata(data_dir, take_id)
    runtime_state = read_json(runtime_state_path(data_dir, take_id))
    result_payload = _result_payload(data_dir, take_id)
    processed_dir = data_dir / "processed" / take_id
    card = build_operations_card(
        take_id=take_id,
        incoming_metadata=incoming_metadata,
        runtime_state=runtime_state,
        result_payload=result_payload,
        processed_dir=processed_dir,
    )
    write_operations_card(processed_dir, card)
    update_operations_index(data_dir, card)
    return card


def _ensure_failed_result(data_dir: Path, take_id: str, message: str) -> None:
    processed_dir = data_dir / "processed" / take_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    result_path = processed_dir / "result.json"
    if result_path.is_file():
        return
    payload = {
        "take_id": take_id,
        "status": "failed",
        "error": message,
        "processed_at": now_iso(),
        "processing_pipeline": {
            "id": "mining_steel_ball_classification_25d",
            "pipeline_family": "25d",
        },
        "objects": [],
        "summary": {"object_count": 0, "decision": "review", "confidence": None},
    }
    write_json(result_path, payload)


def _promote_acquired_to_queued(data_dir: Path, take_id: str) -> None:
    state = read_json(runtime_state_path(data_dir, take_id)) or {}
    if str(state.get("state") or "") == "acquired":
        _mark_state(data_dir, take_id, "queued", {"queued_at": now_iso()})


def _execute_take(settings: ApiSettings, take_id: str, pipeline_id: str) -> tuple[bool, str | None]:
    _mark_state(settings.data_dir, take_id, "processing", {"processing_started_at": now_iso(), "pipeline_id": pipeline_id})
    output_dir = settings.processed_dir / take_id
    if output_dir.exists():
        _remove_dir_with_retry(output_dir)
    try:
        dispatch = dispatch_take_processing(
            settings=settings,
            take_id=take_id,
            pipeline_id=pipeline_id,
            reprocess=True,
            source_id="trispector_ftp",
            acquisition_group_id=None,
            recipe_version_id=None,
            calibration_profile_id=None,
        )
        result_payload = _result_payload(settings.data_dir, take_id)
        status = str((result_payload or {}).get("status") or dispatch.get("status") or "")
        if status.lower() in OK_STATUSES:
            _mark_state(settings.data_dir, take_id, "completed", {"processed_at": now_iso()})
            return True, None
        message = str((result_payload or {}).get("error") or f"Pipeline ended with status={status or 'unknown'}")
        _mark_state(settings.data_dir, take_id, "failed", {"error": message, "processed_at": now_iso()})
        return False, message
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        _mark_state(settings.data_dir, take_id, "failed", {"error": message, "processed_at": now_iso()})
        _ensure_failed_result(settings.data_dir, take_id, message)
        return False, message


def _remove_dir_with_retry(path: Path, retries: int = 5, delay_sec: float = 0.15) -> None:
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(max(0.01, delay_sec))
    if path.exists():
        # Final attempt through temporary rename helps on transient macOS metadata races.
        tmp = path.with_name(f".delete_{path.name}_{int(time.time() * 1000)}")
        try:
            path.rename(tmp)
            shutil.rmtree(tmp)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error is not None:
        raise last_error


def run_iteration(settings: ApiSettings, pipeline_id: str, retry_failed: bool, max_per_iteration: int) -> dict[str, Any]:
    processed = 0
    failed = 0
    cards = 0
    candidates = _list_candidate_take_ids(settings.data_dir, retry_failed=retry_failed)
    for take_id in candidates[: max(1, int(max_per_iteration))]:
        _promote_acquired_to_queued(settings.data_dir, take_id)
        ok, error = _execute_take(settings, take_id, pipeline_id)
        card = _publish_card_and_index(settings.data_dir, take_id)
        cards += 1
        if ok:
            processed += 1
            print(json.dumps({"take_id": take_id, "status": "completed", "superclass": card.get("superclass")}))
        else:
            failed += 1
            print(json.dumps({"take_id": take_id, "status": "failed", "error": error}))
    return {"processed": processed, "failed": failed, "cards_updated": cards, "queued_seen": len(candidates)}


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    settings = _settings(data_dir)
    heartbeat = ProcessHeartbeat(
        RuntimeProcessRegistry(data_dir),
        process_name="run_25d_worker",
        runtime_role="worker",
        pid=os.getpid(),
        interval_seconds=3.0,
        base_metadata={"pipeline_id": args.pipeline_id},
    )
    heartbeat.start()

    totals = {"processed": 0, "failed": 0, "cards_updated": 0, "iterations": 0}
    try:
        while True:
            summary = run_iteration(
                settings,
                pipeline_id=args.pipeline_id,
                retry_failed=bool(args.retry_failed),
                max_per_iteration=max(1, int(args.max_per_iteration)),
            )
            totals["processed"] += int(summary["processed"])
            totals["failed"] += int(summary["failed"])
            totals["cards_updated"] += int(summary["cards_updated"])
            totals["iterations"] += 1
            heartbeat.pulse(
                counters={
                    "processed_total": totals["processed"],
                    "failed_total": totals["failed"],
                    "cards_updated_total": totals["cards_updated"],
                    "iterations": totals["iterations"],
                    "queued_seen_last": int(summary.get("queued_seen") or 0),
                }
            )
            if args.once:
                break
            time.sleep(max(0.1, float(args.poll_interval_sec)))
    except Exception as exc:  # noqa: BLE001
        heartbeat.set_failed(str(exc))
        raise
    finally:
        heartbeat.stop(status="stopped")

    print(json.dumps({"worker": "25d", "pipeline_id": args.pipeline_id, **totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
