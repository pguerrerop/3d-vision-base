from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping


PipelineFamily = Literal["3d", "2d", "25d", "generic"]
ProcessingState = Literal["never_processed", "completed", "running", "failed", "partial"]


@dataclass
class ProcessingSource:
    type: Literal["3d_done_marker", "process_run"]
    path: str | None = None
    run_id: str | None = None
    pipeline_instance_id: str | None = None


@dataclass
class FamilyProcessingSummary:
    family: PipelineFamily
    has_completed_output: bool
    last_run_at: str | None
    status: ProcessingState
    sources: list[ProcessingSource]


def index_path(data_dir: Path) -> Path:
    return data_dir / "processes" / "index" / "runs.json"


def append_process_run_index(
    data_dir: Path,
    *,
    take_id: str | None,
    pipeline_instance_id: str,
    run_id: str,
    pipeline_family: PipelineFamily,
    status: str,
    run_dir: Path,
    created_at: str,
    pipeline_id: str | None = None,
    recipe_version_id: str | None = None,
    config_snapshot_hash: str | None = None,
    source_id: str | None = None,
    acquisition_group_id: str | None = None,
    calibration_profile_id: str | None = None,
    execution_mode: str | None = None,
    parent_run_id: str | None = None,
    parent_take_id: str | None = None,
    partial_rerun_plan_id: str | None = None,
    boundary_stage_id: str | None = None,
    boundary_unit_id: str | None = None,
    selected_unit_id: str | None = None,
    comparison_id: str | None = None,
    recipe_id: str | None = None,
    run_summary: Mapping[str, Any] | None = None,
) -> None:
    if not take_id:
        return
    entry = {
        "take_id": take_id,
        "pipeline_instance_id": pipeline_instance_id,
        "run_id": run_id,
        "pipeline_family": pipeline_family,
        "status": status,
        "created_at": created_at,
        "path": str(run_dir),
        "pipeline_id": pipeline_id,
        "recipe_version_id": recipe_version_id,
        "config_snapshot_hash": config_snapshot_hash,
        "source_id": source_id,
        "acquisition_group_id": acquisition_group_id,
        "calibration_profile_id": calibration_profile_id,
        "execution_mode": execution_mode,
        "parent_run_id": parent_run_id,
        "parent_take_id": parent_take_id,
        "partial_rerun_plan_id": partial_rerun_plan_id,
        "boundary_stage_id": boundary_stage_id,
        "boundary_unit_id": boundary_unit_id,
        "selected_unit_id": selected_unit_id,
        "comparison_id": comparison_id,
        "recipe_id": recipe_id,
        "summary": dict(run_summary) if run_summary else {},
    }
    if _store_enabled(data_dir):
        from vision_3d_acquisition.storage import run_store

        run_store.append_run(data_dir, entry)
    else:
        _append_to_json(data_dir, entry)

    _sync_catalog(data_dir, take_id)


def _append_to_json(data_dir: Path, entry: dict[str, Any]) -> None:
    """The pre-stage-2 path: rewrite the whole document to add one entry.

    Kept reachable through SENSOR_STUDIO_INDEX=off for one release. It is the
    implementation that loses entries when two runs finish at the same time.
    """
    path = index_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_index(path)
    payload.setdefault("entries", []).append(entry)
    payload["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_process_run_index_entry(data_dir: Path, run_id: str, **patch: Any) -> None:
    """Merge patch fields into the run matching run_id, e.g. to backfill
    comparison_id once a comparison is generated after the run was recorded."""
    if _store_enabled(data_dir):
        from vision_3d_acquisition.storage import run_store

        touched_take_ids = set(run_store.update_run(data_dir, run_id, patch))
    else:
        touched_take_ids = _update_in_json(data_dir, run_id, patch)

    for take_id in sorted(touched_take_ids):
        _sync_catalog(data_dir, take_id)


def _update_in_json(data_dir: Path, run_id: str, patch: dict[str, Any]) -> set[str]:
    path = index_path(data_dir)
    payload = _read_index(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return set()
    changed = False
    touched_take_ids: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("run_id") or "") == run_id:
            entry.update(patch)
            changed = True
            if entry.get("take_id"):
                touched_take_ids.add(str(entry["take_id"]))
    if not changed:
        return set()
    payload["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return touched_take_ids


def _store_enabled(data_dir: Path) -> bool:
    """Whether runs live in SQLite. False reverts to the legacy JSON document."""
    from vision_3d_acquisition.storage import catalog_sync  # noqa: WPS433 - avoids an import cycle

    return catalog_sync.index_enabled()


def _sync_catalog(data_dir: Path, take_id: str | None) -> None:
    """Tell the catalog a take's run history moved. Best effort; see catalog_sync."""
    if not take_id:
        return
    from vision_3d_acquisition.storage import catalog_sync  # noqa: WPS433 - avoids an import cycle

    catalog_sync.refresh_take(data_dir, take_id)


def normalize_run_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Backward-compat view of an index entry: fills execution_mode/parent-linkage
    defaults for legacy entries that predate partial-rerun child-run support."""
    normalized = dict(entry)
    normalized.setdefault("execution_mode", None)
    if not normalized.get("execution_mode"):
        normalized["execution_mode"] = "full_run"
    for key in (
        "parent_run_id",
        "parent_take_id",
        "partial_rerun_plan_id",
        "boundary_stage_id",
        "boundary_unit_id",
        "selected_unit_id",
        "comparison_id",
        "recipe_id",
    ):
        normalized.setdefault(key, None)
    summary = normalized.get("summary")
    normalized["summary"] = dict(summary) if isinstance(summary, dict) else {}
    return normalized


def load_process_entries(data_dir: Path) -> list[dict[str, Any]]:
    if _store_enabled(data_dir):
        from vision_3d_acquisition.storage import run_store

        ensure_runs_migrated(data_dir)
        return run_store.load_runs(data_dir)
    return _load_entries_from_json(data_dir)


def process_entries_for_take(data_dir: Path, take_id: str) -> list[dict[str, Any]]:
    if _store_enabled(data_dir):
        from vision_3d_acquisition.storage import run_store

        ensure_runs_migrated(data_dir)
        return run_store.runs_for_take(data_dir, take_id)
    return [item for item in _load_entries_from_json(data_dir) if str(item.get("take_id") or "") == take_id]


def ensure_runs_migrated(data_dir: Path, conn=None) -> int:
    """Move runs.json into the table the first time, then never again.

    Emptiness is the migration flag: once a run is stored, runs.json stops being
    read, so a stale mirror can never overwrite live rows. ``conn`` is passed by
    callers that already hold a transaction.
    """
    from vision_3d_acquisition.storage import run_store

    if not run_store.is_empty(data_dir, conn):
        return 0
    legacy = _load_entries_from_json(data_dir)
    if not legacy:
        return 0
    return run_store.import_entries(data_dir, legacy, conn)


def _load_entries_from_json(data_dir: Path) -> list[dict[str, Any]]:
    path = index_path(data_dir)
    payload = _read_index(path)
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    if entries:
        return [item for item in entries if isinstance(item, dict)]
    return _rebuild_entries(data_dir)


def summarize_take_processing(data_dir: Path, take_id: str, *, has_3d_done: bool, processed_dir: Path) -> list[dict[str, Any]]:
    families: dict[str, FamilyProcessingSummary] = {}

    result_payload: dict[str, Any] | None = None
    result_path = processed_dir / "result.json"
    if result_path.is_file():
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                result_payload = parsed
        except Exception:
            result_payload = None
    done_family = str(((result_payload or {}).get("processing_pipeline") or {}).get("pipeline_family") or "3d")
    if done_family not in {"3d", "25d"}:
        done_family = "3d"

    if has_3d_done:
        done_marker = processed_dir / "DONE"
        families[done_family] = FamilyProcessingSummary(
            family=done_family,  # type: ignore[arg-type]
            has_completed_output=True,
            last_run_at=datetime.fromtimestamp(done_marker.stat().st_mtime, tz=UTC).isoformat() if done_marker.exists() else None,
            status="completed",
            sources=[ProcessingSource(type="3d_done_marker", path=str(done_marker))],
        )
    for family in ("3d", "25d"):
        if family in families:
            continue
        families[family] = FamilyProcessingSummary(
            family=family,  # type: ignore[arg-type]
            has_completed_output=False,
            last_run_at=None,
            status="never_processed",
            sources=[],
        )

    entries = [item for item in load_process_entries(data_dir) if str(item.get("take_id")) == take_id]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        family = str(entry.get("pipeline_family") or "generic")
        grouped.setdefault(family, []).append(entry)

    for family in ("2d", "generic"):
        items = sorted(grouped.get(family, []), key=lambda item: str(item.get("created_at") or ""), reverse=True)
        if not items:
            families[family] = FamilyProcessingSummary(family=family, has_completed_output=False, last_run_at=None, status="never_processed", sources=[])
            continue
        latest = items[0]
        latest_status = str(latest.get("status") or "completed")
        resolved = "completed" if latest_status in {"success", "warning", "completed"} else "failed" if latest_status == "failed" else "partial"
        families[family] = FamilyProcessingSummary(
            family=family,  # type: ignore[arg-type]
            has_completed_output=resolved == "completed",
            last_run_at=str(latest.get("created_at") or ""),
            status=resolved,  # type: ignore[arg-type]
            sources=[
                ProcessingSource(
                    type="process_run",
                    path=str(item.get("path") or "") or None,
                    run_id=str(item.get("run_id") or "") or None,
                    pipeline_instance_id=str(item.get("pipeline_instance_id") or "") or None,
                )
                for item in items[:3]
            ],
        )

    return [
        {
            "family": item.family,
            "hasCompletedOutput": item.has_completed_output,
            "lastRunAt": item.last_run_at,
            "status": item.status,
            "sources": [source.__dict__ for source in item.sources],
        }
        for item in families.values()
    ]


def _read_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": []}
    return payload if isinstance(payload, dict) else {"entries": []}


def _rebuild_entries(data_dir: Path) -> list[dict[str, Any]]:
    instances_dir = data_dir / "processes" / "instances"
    if not instances_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for instance_file in instances_dir.glob("*.json"):
        try:
            payload = json.loads(instance_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        instance_id = str(payload.get("id") or "")
        for run in payload.get("execution_history") or []:
            if not isinstance(run, dict):
                continue
            summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
            take_id = summary.get("take_id")
            if not take_id:
                continue
            entries.append(
                {
                    "take_id": str(take_id),
                    "pipeline_instance_id": instance_id,
                    "run_id": str(run.get("run_id") or ""),
                    "pipeline_family": "2d",
                    "status": str(run.get("status") or "completed"),
                    "created_at": str(run.get("timestamp") or ""),
                    "path": None,
                }
            )
    return entries
