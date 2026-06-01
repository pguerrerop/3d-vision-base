from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vision_3d_acquisition.datasets import DatasetService


@dataclass
class TakeRefMatch:
    source_ref: str
    matched_take_id: str | None
    status: str
    candidates: list[str]


def build_take_index(service: DatasetService, dataset_id: str, session_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
    rows = service.iter_dataset_takes(dataset_id)
    allowed = set(session_ids or [])
    index: dict[str, dict[str, Any]] = {}
    for sid, take_id, meta in rows:
        if allowed and sid not in allowed:
            continue
        index[take_id] = {"session_id": sid, "metadata": meta}
    return index


def resolve_reference(ref: str, take_index: dict[str, dict[str, Any]]) -> TakeRefMatch:
    value = str(ref or "").strip()
    if not value:
        return TakeRefMatch(source_ref=value, matched_take_id=None, status="empty", candidates=[])
    if value in take_index:
        return TakeRefMatch(source_ref=value, matched_take_id=value, status="exact", candidates=[value])
    suffix = value.zfill(3) if value.isdigit() else value
    candidates = [take_id for take_id in take_index if take_id.endswith(f"_{suffix}") or take_id.endswith(suffix)]
    if len(candidates) == 1:
        return TakeRefMatch(source_ref=value, matched_take_id=candidates[0], status="abbreviated", candidates=candidates)
    if len(candidates) > 1:
        return TakeRefMatch(source_ref=value, matched_take_id=None, status="ambiguous", candidates=sorted(candidates))
    return TakeRefMatch(source_ref=value, matched_take_id=None, status="unresolved", candidates=[])
