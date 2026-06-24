from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = [
    "source_type",
    "source_filename",
    "source_detected_format",
    "source_byte_size",
    "source_sha256",
    "source_row_count",
    "source_column_count",
    "physical_object_id",
    "take_id",
    "image_ref",
    "source_session_id",
    "source_row",
    "raw_operator_label",
    "raw_label",
    "label",
    "normalized_class",
    "superclass",
    "label_policy",
    "review_required",
    "include",
    "default_trainable",
    "d1_mm",
    "d2_mm",
    "d3_mm",
    "annotation_confidence",
    "needs_review",
    "source_row_index",
    "resolution_method",
    "created_from_range",
    "normalization_source",
    "normalization_version",
    "label_schema_id",
    "extra_fields",
]


def build_canonical_manifest(run_dir: Path, rows: list[dict[str, Any]], diagnostics: dict[str, list[dict[str, Any]]], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    out_dir = run_dir / "canonical_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_provenance = provenance if isinstance(provenance, dict) else {}

    with (out_dir / "label_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            merged = {
                "source_type": source_provenance.get("source_type"),
                "source_filename": source_provenance.get("filename"),
                "source_detected_format": source_provenance.get("detected_format"),
                "source_byte_size": source_provenance.get("byte_size"),
                "source_sha256": source_provenance.get("sha256"),
                "source_row_count": source_provenance.get("row_count"),
                "source_column_count": source_provenance.get("column_count"),
                **row,
            }
            writer.writerow({k: merged.get(k) for k in MANIFEST_COLUMNS})

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
            "superclass": row.get("superclass"),
            "d1_mm": row.get("d1_mm"),
            "d2_mm": row.get("d2_mm"),
            "d3_mm": row.get("d3_mm"),
        })
        object_rows[obj]["take_count"] += 1

    with (out_dir / "object_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["physical_object_id", "take_count", "normalized_class", "superclass", "d1_mm", "d2_mm", "d3_mm"]
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
        "source_provenance": source_provenance,
        "unresolved_refs": len(diagnostics.get("unresolved_take_refs", [])),
        "ambiguous_refs": len(diagnostics.get("ambiguous_take_refs", [])),
        "duplicate_assignments": len(diagnostics.get("duplicate_take_assignments", [])),
        "unlabeled_pool": len(diagnostics.get("unlabeled_pool", [])),
        "object_conflicts": len(diagnostics.get("object_conflicts", [])),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "source_provenance.json").write_text(json.dumps(source_provenance, ensure_ascii=False, indent=2), encoding="utf-8")

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
            "source_provenance_json": str(out_dir / "source_provenance.json"),
        },
        "validation_report": validation_report,
    }
