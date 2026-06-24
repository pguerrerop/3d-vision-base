from __future__ import annotations

from collections import defaultdict
from typing import Any


def assign_physical_object_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_physical_object_grouping(rows)["rows"]


def build_physical_object_grouping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: list[dict[str, Any]] = []
    row_to_obj: dict[str, str] = {}
    counter = 0
    missing_object_id_rows = 0
    object_semantics: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        explicit_object_id = str(row.get("physical_object_id") or row.get("object_id") or "").strip()
        source_row_value = str(row.get("source_row") or row.get("source_row_index") or "").strip()
        if explicit_object_id:
            physical_object_id = explicit_object_id
        else:
            if not source_row_value:
                source_row_value = str(row.get("source_row_index") or len(grouped) + 1)
            if source_row_value not in row_to_obj:
                counter += 1
                row_to_obj[source_row_value] = f"object_{counter:06d}"
            physical_object_id = row_to_obj[source_row_value]
            missing_object_id_rows += 1

        next_row = {**row, "physical_object_id": physical_object_id}
        grouped.append(next_row)

        semantic_bucket = object_semantics.setdefault(physical_object_id, defaultdict(set))
        for field in ("label", "normalized_class", "superclass", "d1_mm", "d2_mm", "d3_mm"):
            value = str(next_row.get(field) or "").strip()
            if value:
                semantic_bucket[field].add(value)

    conflicts: list[dict[str, Any]] = []
    object_distribution_by_class: dict[str, int] = defaultdict(int)
    object_distribution_by_superclass: dict[str, int] = defaultdict(int)
    for physical_object_id, bucket in object_semantics.items():
        field_conflicts = {
            field: sorted(values)
            for field, values in bucket.items()
            if len(values) > 1
        }
        if field_conflicts:
            conflicts.append({
                "physical_object_id": physical_object_id,
                "conflicts": field_conflicts,
            })
        normalized_class = next(iter(bucket.get("normalized_class") or []), "")
        superclass = next(iter(bucket.get("superclass") or []), "")
        object_distribution_by_class[normalized_class or "UNLABELED"] += 1
        object_distribution_by_superclass[superclass or "UNKNOWN"] += 1

    warnings: list[str] = []
    if missing_object_id_rows:
        warnings.append("Some rows are missing object_id/physical_object_id; leakage-safe splitting is weaker for inferred groups.")

    return {
        "rows": grouped,
        "summary": {
            "physical_object_count": len(object_semantics),
            "take_count": len(grouped),
            "rows_with_object_id": len(grouped) - missing_object_id_rows,
            "rows_missing_object_id": missing_object_id_rows,
            "conflict_count": len(conflicts),
            "distribution_by_normalized_class": dict(sorted(object_distribution_by_class.items(), key=lambda item: item[0])),
            "distribution_by_superclass": dict(sorted(object_distribution_by_superclass.items(), key=lambda item: item[0])),
            "warnings": warnings,
        },
        "conflicts": conflicts,
    }
