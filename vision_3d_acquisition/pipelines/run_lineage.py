from __future__ import annotations

from typing import Any

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.processing.status_index import (
    load_process_entries,
    normalize_run_entry,
    process_entries_for_take,
    update_process_run_index_entry,
)


def enrich_run_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalizes a raw index entry and adds a computed run_type."""
    normalized = normalize_run_entry(entry)
    normalized["run_type"] = "partial_rerun" if normalized.get("parent_run_id") else "full_run"
    return normalized


def list_pipeline_runs(
    settings: ApiSettings,
    pipeline_id: str,
    *,
    take_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    entries = process_entries_for_take(settings.data_dir, take_id) if take_id else load_process_entries(settings.data_dir)
    filtered = [entry for entry in entries if str(entry.get("pipeline_id") or "") == pipeline_id]
    ordered = sorted(filtered, key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return [enrich_run_entry(entry) for entry in ordered[:limit]]


def resolve_run_parent(settings: ApiSettings, pipeline_id: str, run_id: str) -> dict[str, Any] | None:
    entries = [entry for entry in load_process_entries(settings.data_dir) if str(entry.get("pipeline_id") or "") == pipeline_id]
    current = next((entry for entry in entries if str(entry.get("run_id") or "") == run_id), None)
    if current is None:
        return None
    parent_run_id = current.get("parent_run_id")
    if not parent_run_id:
        return None
    parent = next((entry for entry in entries if str(entry.get("run_id") or "") == str(parent_run_id)), None)
    if parent is None:
        return None
    return enrich_run_entry(parent)


def resolve_run_children(settings: ApiSettings, pipeline_id: str, run_id: str) -> list[dict[str, Any]]:
    entries = [entry for entry in load_process_entries(settings.data_dir) if str(entry.get("pipeline_id") or "") == pipeline_id]
    children = [entry for entry in entries if str(entry.get("parent_run_id") or "") == run_id]
    ordered = sorted(children, key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return [enrich_run_entry(entry) for entry in ordered]


def resolve_run_comparison_id(settings: ApiSettings, pipeline_id: str, run_id: str) -> str | None:
    entries = [entry for entry in load_process_entries(settings.data_dir) if str(entry.get("pipeline_id") or "") == pipeline_id]
    current = next((entry for entry in entries if str(entry.get("run_id") or "") == run_id), None)
    if current is None:
        return None
    comparison_id = current.get("comparison_id")
    return str(comparison_id) if comparison_id else None


def generate_run_comparison_to_parent(settings: ApiSettings, pipeline_id: str, run_id: str) -> dict[str, Any]:
    """Generates (or returns the existing) parent-vs-child comparison for a partial
    child run, backfilling comparison_id onto its index entry. Reuses the same
    compare_pipeline_sources() helper that execute_partial_rerun() calls at execute time,
    so this is the single source of comparison-generation logic for both paths."""
    from vision_3d_acquisition.pipelines.comparison import compare_pipeline_sources

    entries = [entry for entry in load_process_entries(settings.data_dir) if str(entry.get("pipeline_id") or "") == pipeline_id]
    current = next((entry for entry in entries if str(entry.get("run_id") or "") == run_id), None)
    if current is None:
        raise KeyError(f"Run not found: {run_id}")
    parent_run_id = current.get("parent_run_id")
    if not parent_run_id:
        raise ValueError(f"Run {run_id} has no parent run to compare against.")
    existing_comparison_id = current.get("comparison_id")
    if existing_comparison_id:
        from vision_3d_acquisition.pipelines.comparison import get_pipeline_comparison

        return get_pipeline_comparison(settings, pipeline_id, str(existing_comparison_id))
    take_id = str(current.get("take_id") or current.get("parent_take_id") or "")
    if not take_id:
        raise ValueError(f"Run {run_id} is missing a take_id required to generate a comparison.")
    comparison = compare_pipeline_sources(
        settings,
        pipeline_id=pipeline_id,
        left={"type": "run", "take_id": take_id, "run_id": str(parent_run_id)},
        right={"type": "run", "take_id": take_id, "run_id": run_id},
    )
    comparison_id = str(comparison.get("comparison_id") or "") or None
    if comparison_id:
        update_process_run_index_entry(settings.data_dir, run_id, comparison_id=comparison_id)
    return comparison
