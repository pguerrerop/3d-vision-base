from __future__ import annotations

from typing import Any


def assign_physical_object_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    row_to_obj: dict[int, str] = {}
    counter = 0
    for row in rows:
        source_row_index = int(row.get("source_row_index") or 0)
        if source_row_index not in row_to_obj:
            counter += 1
            row_to_obj[source_row_index] = f"object_{counter:06d}"
        grouped.append({**row, "physical_object_id": row_to_obj[source_row_index]})
    return grouped
