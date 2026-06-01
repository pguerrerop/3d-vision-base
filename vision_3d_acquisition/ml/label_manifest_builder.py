from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = [
    "physical_object_id",
    "take_id",
    "source_session_id",
    "raw_operator_label",
    "normalized_class",
    "superclass",
    "d1_mm",
    "d2_mm",
    "d3_mm",
    "annotation_confidence",
    "needs_review",
    "source_row_index",
    "resolution_method",
    "created_from_range",
    "label_schema_id",
]


def build_canonical_manifest(run_dir: Path, rows: list[dict[str, Any]], diagnostics: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out_dir = run_dir / "canonical_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "label_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in MANIFEST_COLUMNS})

    with (out_dir / "label_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    object_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        obj = str(row.get("physical_object_id") or "")
        if not obj:
            continue
        object_rows.setdefault(obj, {
            "physical_object_id": obj,
            "take_count": 0,
            "normalized_class": row.get("normalized_class"),
            "d1_mm": row.get("d1_mm"),
            "d2_mm": row.get("d2_mm"),
            "d3_mm": row.get("d3_mm"),
        })
        object_rows[obj]["take_count"] += 1

    with (out_dir / "object_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["physical_object_id", "take_count", "normalized_class", "d1_mm", "d2_mm", "d3_mm"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(object_rows.values(), key=lambda item: str(item["physical_object_id"])):
            writer.writerow(row)

    for key in ("unresolved_take_refs", "ambiguous_take_refs", "duplicate_take_assignments", "unlabeled_pool"):
        rows_for_key = diagnostics.get(key, [])
        path = out_dir / f"{key}.csv"
        if rows_for_key:
            fieldnames = sorted({k for row in rows_for_key for k in row.keys()})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows_for_key:
                    writer.writerow(row)
        else:
            path.write_text("", encoding="utf-8")

    validation_report = {
        "manifest_count": len(rows),
        "unresolved_refs": len(diagnostics.get("unresolved_take_refs", [])),
        "ambiguous_refs": len(diagnostics.get("ambiguous_take_refs", [])),
        "duplicate_assignments": len(diagnostics.get("duplicate_take_assignments", [])),
        "unlabeled_pool": len(diagnostics.get("unlabeled_pool", [])),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "files": {
            "label_manifest_csv": str(out_dir / "label_manifest.csv"),
            "label_manifest_jsonl": str(out_dir / "label_manifest.jsonl"),
            "object_groups_csv": str(out_dir / "object_groups.csv"),
            "unresolved_take_refs_csv": str(out_dir / "unresolved_take_refs.csv"),
            "ambiguous_take_refs_csv": str(out_dir / "ambiguous_take_refs.csv"),
            "duplicate_take_assignments_csv": str(out_dir / "duplicate_take_assignments.csv"),
            "unlabeled_pool_csv": str(out_dir / "unlabeled_pool.csv"),
            "validation_report_json": str(out_dir / "validation_report.json"),
        },
        "validation_report": validation_report,
    }
