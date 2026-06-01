from __future__ import annotations

import re
from typing import Any


def _to_int(value: Any) -> int | None:
    raw = str(value or "").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def expand_row_references(row: dict[str, Any]) -> list[str]:
    base = str(row.get("image_ref") or row.get("image") or row.get("take") or "").strip()
    from_v = _to_int(row.get("from") or row.get("start") or row.get("first"))
    to_v = _to_int(row.get("to") or row.get("end") or row.get("last"))
    if from_v is not None and to_v is not None and to_v >= from_v:
        return [str(i) for i in range(from_v, to_v + 1)]
    dots = re.match(r"^\s*(\d+)\s*(?:\.\.\.|-+)\s*(\d+)\s*$", base)
    if dots:
        a = int(dots.group(1))
        b = int(dots.group(2))
        if b >= a:
            return [str(i) for i in range(a, b + 1)]
    if base:
        return [base]
    return []
