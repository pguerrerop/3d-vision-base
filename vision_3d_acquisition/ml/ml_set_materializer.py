from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vision_3d_acquisition.datasets import DatasetService


def materialize_ml_set(*, data_dir: Path, ml_set_id: str, dataset_id: str, manifest_rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    out_dir = data_dir / "ml_sets" / ml_set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    include_unlabeled = bool(policy.get("include_unlabeled", False))
    include_review_required = bool(policy.get("include_review_required", False))
    dataset_service = DatasetService(data_dir)
    ml_set = dataset_service.get_ml_set(dataset_id, ml_set_id)
    created_or_updated = "updated" if ml_set is not None else "created"
    if ml_set is None:
        dataset_service.create_ml_set(
            dataset_id=dataset_id,
            ml_set_id=ml_set_id,
            name=ml_set_id,
            task_type="classification",
        )
        ml_set = dataset_service.get_ml_set(dataset_id, ml_set_id) or {}
    else:
        ml_set = dict(ml_set)

    filtered: list[dict[str, Any]] = []
    trainable: list[dict[str, Any]] = []
    excluded_pool: list[dict[str, Any]] = []
    for row in manifest_rows:
        normalized = str(row.get("normalized_class") or "")
        filtered.append(row)
        include_row = bool(row.get("include", True))
        review_required = bool(row.get("review_required", row.get("needs_review", False)))
        label_policy = str(row.get("label_policy") or "").strip().lower()
        default_trainable = bool(
            row.get(
                "default_trainable",
                include_row and label_policy == "include" and not review_required,
            )
        )
        effective_trainable = default_trainable
        if review_required and include_review_required and include_row and label_policy not in {"exclude", "reference"}:
            effective_trainable = True
        if label_policy in {"exclude", "reference"}:
            effective_trainable = False
        if not normalized and not include_unlabeled:
            effective_trainable = False
        if normalized == "REVIEW_REQUIRED" and not include_review_required:
            effective_trainable = False
        if not include_row:
            effective_trainable = False
        if effective_trainable:
            trainable.append(row)
        else:
            excluded_pool.append(row)
        dataset_service.add_take_to_ml_set(
            dataset_id=dataset_id,
            ml_set_id=ml_set_id,
            take_id=str(row.get("take_id") or ""),
            split="unassigned",
            physical_object_id=str(row.get("physical_object_id") or "") or None,
            include=include_row,
            default_trainable=default_trainable,
            trainable=effective_trainable,
            notes=str(row.get("label_policy") or "") or None,
            expected_label=str(row.get("label") or "") or None,
            expected_class=str(row.get("normalized_class") or "") or None,
            expected_subclass=str(row.get("superclass") or "") or None,
            measurements_mm={
                key: float(row[key])
                for key in ("d1_mm", "d2_mm", "d3_mm")
                if str(row.get(key) or "").strip()
            } or None,
            raw_label=str(row.get("raw_label") or row.get("raw_operator_label") or "") or None,
            label_policy=str(row.get("label_policy") or "") or None,
            review_required=bool(row.get("review_required", row.get("needs_review", False))),
            normalization_version=str(row.get("normalization_version") or row.get("label_schema_id") or "") or None,
            source_row=str(row.get("source_row") or row.get("source_row_index") or "") or None,
            extra_fields=row.get("extra_fields") if isinstance(row.get("extra_fields"), dict) else None,
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trainable:
        grouped[str(row.get("physical_object_id") or "")].append(row)

    split_rows: list[dict[str, Any]] = []
    for obj_id, rows in grouped.items():
        for row in rows:
            split_rows.append({"take_id": row.get("take_id"), "physical_object_id": obj_id, "split": "unassigned"})

    leakage = defaultdict(set)
    for row in split_rows:
        leakage[str(row["physical_object_id"])].add(str(row["split"]))
    bad_groups = [obj for obj, splits in leakage.items() if len(splits) > 1]

    class_dist: dict[str, int] = defaultdict(int)
    for row in trainable:
        class_dist[str(row.get("normalized_class") or "UNKNOWN")] += 1

    obj_dist = {"unassigned": len({str(row.get("physical_object_id") or "") for row in trainable if str(row.get("physical_object_id") or "")})}

    materialized_payload = {
        "id": ml_set_id,
        "dataset_id": dataset_id,
        "name": str(ml_set.get("name") or ml_set_id),
        "immutable": True,
        "manifest_count": len(filtered),
        "trainable_count": len(trainable),
        "membership_mode": "physical_object_id",
        "policy": policy,
    }
    (out_dir / "ml_set.json").write_text(json.dumps(materialized_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "label_schema.json").write_text(json.dumps({"id": "mining_balls_labels_v1"}, ensure_ascii=False, indent=2), encoding="utf-8")

    if filtered:
        with (out_dir / "label_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = sorted({k for row in filtered for k in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in filtered:
                writer.writerow(row)

    with (out_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["take_id", "physical_object_id", "split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in split_rows:
            writer.writerow(row)

    tasks = policy.get("tasks") or ["mining_ball_condition_v1"]
    (out_dir / "tasks.json").write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "class_distribution.json").write_text(json.dumps(class_dist, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "object_group_distribution.json").write_text(json.dumps(obj_dist, ensure_ascii=False, indent=2), encoding="utf-8")

    validation_report = {
        "ok": len(bad_groups) == 0,
        "group_leakage_count": len(bad_groups),
        "leaked_object_ids": bad_groups,
        "kept_in_ml_set_count": sum(1 for row in filtered if bool(row.get("include", True))),
        "default_trainable_count": sum(1 for row in filtered if bool(row.get("default_trainable", False))),
        "excluded_count": len(excluded_pool),
        "trainable_count": len(trainable),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# ML Set\n\nDeterministic materialization from canonical manifest.\n",
        encoding="utf-8",
    )

    memberships = dataset_service.list_ml_set_memberships(dataset_id=dataset_id, ml_set_id=ml_set_id)
    membership_summary = dataset_service.summarize_ml_set_memberships(ml_set, memberships)
    source_type = next((str(row.get("source_type") or "").strip() for row in filtered if str(row.get("source_type") or "").strip()), "")
    source_filename = next((str(row.get("source_filename") or "").strip() for row in filtered if str(row.get("source_filename") or "").strip()), "")
    source_sha256 = next((str(row.get("source_sha256") or "").strip() for row in filtered if str(row.get("source_sha256") or "").strip()), "")
    normalization_version = next((str(row.get("normalization_version") or row.get("label_schema_id") or "").strip() for row in filtered if str(row.get("normalization_version") or row.get("label_schema_id") or "").strip()), "mining_balls_labels_v1")
    dataset_ml_set_path = data_dir / "datasets" / f"dataset_{dataset_id}" / "ml_sets" / f"ml_set_{ml_set_id}"
    persisted_ml_set = {
        **ml_set,
        "updated_at": ml_set.get("updated_at") or materialized_payload.get("updated_at"),
        "membership_mode": membership_summary["membership_mode"],
        "membership_count": membership_summary["membership_count"],
        "physical_object_count": membership_summary["physical_object_count"],
        "default_trainable_rows": membership_summary["default_trainable_rows"],
        "default_trainable_objects": membership_summary["default_trainable_objects"],
        "review_required_rows": membership_summary["review_required_rows"],
        "review_required_objects": membership_summary["review_required_objects"],
        "split_status": membership_summary["split_status"],
        "normalization_version": normalization_version,
        "provenance": {
            "source_type": source_type or None,
            "filename": source_filename or None,
            "sha256": source_sha256 or None,
        },
        "membership": {
            **((ml_set.get("membership") or {}) if isinstance(ml_set.get("membership"), dict) else {}),
            "mode": "physical_object_id",
        },
        "semantics": {
            **((ml_set.get("semantics") or {}) if isinstance(ml_set.get("semantics"), dict) else {}),
            "source_type": source_type or None,
            "source_filename": source_filename or None,
            "source_sha256": source_sha256 or None,
            "normalization_version": normalization_version,
            "split_status": membership_summary["split_status"],
            "membership_mode": membership_summary["membership_mode"],
        },
    }
    (dataset_ml_set_path / "ml_set.json").write_text(json.dumps(persisted_ml_set, ensure_ascii=False, indent=2), encoding="utf-8")

    memberships_path = dataset_ml_set_path / "memberships.json"
    ml_set_json_path = dataset_ml_set_path / "ml_set.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not ml_set_json_path.is_file():
        errors.append("ml_set.json was not written.")
    if not memberships_path.is_file():
        errors.append("memberships.json was not written.")
    if not memberships:
        errors.append("No memberships were persisted.")
    if membership_summary["membership_mode"] == "physical_object_id" and membership_summary["physical_object_count"] == 0:
        errors.append("physical_object_id was not persisted for object-aware materialization.")
    if sum(1 for row in memberships if "default_trainable" in row and "review_required" in row) != len(memberships):
        errors.append("default_trainable/review_required flags were not fully persisted.")
    if errors:
        return {
            "ok": False,
            "dataset_id": dataset_id,
            "ml_set_id": ml_set_id,
            "ml_set_name": str(persisted_ml_set.get("name") or ml_set_id),
            "created_or_updated": created_or_updated,
            "ml_set_path": str(ml_set_json_path),
            "memberships_path": str(memberships_path),
            "membership_count": membership_summary["membership_count"],
            "physical_object_count": membership_summary["physical_object_count"],
            "default_trainable_rows": membership_summary["default_trainable_rows"],
            "default_trainable_objects": membership_summary["default_trainable_objects"],
            "review_required_rows": membership_summary["review_required_rows"],
            "review_required_objects": membership_summary["review_required_objects"],
            "split_status": membership_summary["split_status"],
            "warnings": warnings,
            "errors": errors,
            "output_dir": str(out_dir),
            "validation_report": validation_report,
        }
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "ml_set_id": ml_set_id,
        "ml_set_name": str(persisted_ml_set.get("name") or ml_set_id),
        "created_or_updated": created_or_updated,
        "ml_set_path": str(ml_set_json_path),
        "memberships_path": str(memberships_path),
        "membership_count": membership_summary["membership_count"],
        "physical_object_count": membership_summary["physical_object_count"],
        "default_trainable_rows": membership_summary["default_trainable_rows"],
        "default_trainable_objects": membership_summary["default_trainable_objects"],
        "review_required_rows": membership_summary["review_required_rows"],
        "review_required_objects": membership_summary["review_required_objects"],
        "split_status": membership_summary["split_status"],
        "warnings": warnings,
        "errors": errors,
        "output_dir": str(out_dir),
        "validation_report": validation_report,
    }
