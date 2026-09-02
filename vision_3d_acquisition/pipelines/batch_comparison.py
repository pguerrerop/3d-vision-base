from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.comparison import compare_pipeline_sources

# Runs a single pairwise comparison (compare_pipeline_sources) once per take and
# aggregates the results, so a parameter change can be evaluated across a dataset
# instead of one take at a time. See docs/contract_recipe_compare_graph_workflow.md.

MOST_CHANGED_UNITS_LIMIT = 10


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _batch_root(settings: ApiSettings, batch_comparison_id: str) -> Path:
    root = settings.data_dir / "batch_comparisons" / batch_comparison_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _batch_index_path(settings: ApiSettings) -> Path:
    root = settings.data_dir / "batch_comparisons"
    root.mkdir(parents=True, exist_ok=True)
    return root / "index.json"


def _load_batch_index(settings: ApiSettings) -> list[dict[str, Any]]:
    path = _batch_index_path(settings)
    if not path.is_file():
        return []
    payload = _read_json(path)
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        return []
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def _write_batch_index(settings: ApiSettings, entries: list[dict[str, Any]]) -> None:
    path = _batch_index_path(settings)
    path.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")


def _append_batch_index(settings: ApiSettings, entry: dict[str, Any]) -> None:
    entries = _load_batch_index(settings)
    batch_comparison_id = str(entry.get("batch_comparison_id") or "")
    entries = [item for item in entries if str(item.get("batch_comparison_id") or "") != batch_comparison_id]
    entries.insert(0, entry)
    _write_batch_index(settings, entries[:200])


def _fill_take_id(source: Mapping[str, Any], take_id: str) -> dict[str, Any]:
    """Per-take source spec: a "run" source is resolved against each take in turn
    (an explicit take_id in the template always wins, so left/right can still pin
    a specific take when useful); "recipe" and "current" sources describe a fixed
    parameter set with no take of their own, so they pass through unchanged and get
    evaluated against every take in the batch."""
    filled = dict(source)
    if str(filled.get("type") or "current") == "run" and not filled.get("take_id"):
        filled["take_id"] = take_id
    return filled


def _aggregate(per_take: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(per_take)
    if count == 0:
        return {
            "compared_take_count": 0,
            "classification_changed_count": 0,
            "classification_changed_fraction": None,
            "object_count_changed_count": 0,
            "warnings_changed_count": 0,
            "takes_with_mask_changes_count": 0,
            "average_iou": None,
            "min_iou": None,
            "max_iou": None,
            "most_changed_units": [],
        }

    classification_changed_count = 0
    object_count_changed_count = 0
    warnings_changed_count = 0
    takes_with_mask_changes_count = 0
    iou_values: list[float] = []
    unit_change_counter: Counter[str] = Counter()

    for entry in per_take:
        summary = entry.get("summary") if isinstance(entry.get("summary"), Mapping) else {}
        if summary.get("classification_changed"):
            classification_changed_count += 1
        if summary.get("object_count_changed"):
            object_count_changed_count += 1
        if summary.get("warnings_changed"):
            warnings_changed_count += 1
        if int(summary.get("mask_artifacts_changed") or 0) > 0:
            takes_with_mask_changes_count += 1
        average_iou = summary.get("average_iou")
        if isinstance(average_iou, (int, float)):
            iou_values.append(float(average_iou))
        for unit_id in summary.get("key_affected_units") or []:
            if isinstance(unit_id, str) and unit_id:
                unit_change_counter[unit_id] += 1

    most_changed_units = [
        {"unit_id": unit_id, "changed_take_count": changed_count, "changed_take_fraction": changed_count / count}
        for unit_id, changed_count in unit_change_counter.most_common(MOST_CHANGED_UNITS_LIMIT)
    ]

    return {
        "compared_take_count": count,
        "classification_changed_count": classification_changed_count,
        "classification_changed_fraction": classification_changed_count / count,
        "object_count_changed_count": object_count_changed_count,
        "object_count_changed_fraction": object_count_changed_count / count,
        "warnings_changed_count": warnings_changed_count,
        "warnings_changed_fraction": warnings_changed_count / count,
        "takes_with_mask_changes_count": takes_with_mask_changes_count,
        "takes_with_mask_changes_fraction": takes_with_mask_changes_count / count,
        "average_iou": float(sum(iou_values) / len(iou_values)) if iou_values else None,
        "min_iou": min(iou_values) if iou_values else None,
        "max_iou": max(iou_values) if iou_values else None,
        "most_changed_units": most_changed_units,
    }


def compare_pipeline_sources_batch(
    settings: ApiSettings,
    *,
    pipeline_id: str,
    take_ids: list[str],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    batch_comparison_id = f"batchcmp_{uuid4().hex[:12]}"
    deduped_take_ids = list(dict.fromkeys(str(item) for item in take_ids if str(item)))

    per_take: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for take_id in deduped_take_ids:
        left_source = _fill_take_id(left, take_id)
        right_source = _fill_take_id(right, take_id)
        try:
            result = compare_pipeline_sources(settings, pipeline_id=pipeline_id, left=left_source, right=right_source)
        except (KeyError, ValueError) as exc:
            # str(KeyError(...)) reprs its argument (adds quotes); unwrap it so
            # the batch error list reads like a normal message.
            message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            errors.append({"take_id": take_id, "error": message})
            continue
        per_take.append(
            {
                "take_id": take_id,
                "comparison_id": result.get("comparison_id"),
                "summary": result.get("summary"),
            }
        )

    payload = {
        "batch_comparison_id": batch_comparison_id,
        "pipeline_id": pipeline_id,
        "created_at": _now_iso(),
        "left": dict(left),
        "right": dict(right),
        "requested_take_count": len(deduped_take_ids),
        "compared_take_count": len(per_take),
        "error_take_count": len(errors),
        "errors": errors,
        "aggregate": _aggregate(per_take),
        "per_take": per_take,
    }

    root = _batch_root(settings, batch_comparison_id)
    (root / "batch_comparison.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    _append_batch_index(
        settings,
        {
            "batch_comparison_id": batch_comparison_id,
            "pipeline_id": pipeline_id,
            "created_at": payload["created_at"],
            "requested_take_count": payload["requested_take_count"],
            "compared_take_count": payload["compared_take_count"],
            "error_take_count": payload["error_take_count"],
            "aggregate": payload["aggregate"],
        },
    )
    return payload


def list_pipeline_batch_comparisons(
    settings: ApiSettings,
    pipeline_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    entries = [item for item in _load_batch_index(settings) if str(item.get("pipeline_id") or "") == pipeline_id]
    return entries[: max(1, min(limit, 100))]


def get_pipeline_batch_comparison(settings: ApiSettings, pipeline_id: str, batch_comparison_id: str) -> dict[str, Any]:
    path = settings.data_dir / "batch_comparisons" / batch_comparison_id / "batch_comparison.json"
    if not path.is_file():
        raise KeyError(f"Unknown batch comparison: {batch_comparison_id}")
    payload = _read_json(path)
    if payload is None:
        raise KeyError(f"Unreadable batch comparison: {batch_comparison_id}")
    if str(payload.get("pipeline_id") or "") != pipeline_id:
        raise KeyError(f"Batch comparison {batch_comparison_id} does not belong to pipeline {pipeline_id}")
    return payload
