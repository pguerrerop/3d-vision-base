from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def materialize_ml_set(*, data_dir: Path, ml_set_id: str, dataset_id: str, manifest_rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    out_dir = data_dir / "ml_sets" / ml_set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    include_unlabeled = bool(policy.get("include_unlabeled", False))
    include_review_required = bool(policy.get("include_review_required", False))

    filtered: list[dict[str, Any]] = []
    unlabeled_pool: list[dict[str, Any]] = []
    for row in manifest_rows:
        normalized = str(row.get("normalized_class") or "")
        if normalized == "REVIEW_REQUIRED" and not include_review_required:
            unlabeled_pool.append(row)
            continue
        if not normalized and not include_unlabeled:
            unlabeled_pool.append(row)
            continue
        filtered.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        grouped[str(row.get("physical_object_id") or "")].append(row)

    object_ids = [k for k in grouped.keys() if k]
    random.Random(int(policy.get("seed", 42))).shuffle(object_ids)
    ratios = policy.get("split_ratios") or {"train": 0.7, "validation": 0.15, "test": 0.15}
    n = len(object_ids)
    n_train = int(n * float(ratios.get("train", 0.7)))
    n_val = int(n * float(ratios.get("validation", 0.15)))
    train_ids = set(object_ids[:n_train])
    val_ids = set(object_ids[n_train:n_train + n_val])
    test_ids = set(object_ids[n_train + n_val:])

    split_rows: list[dict[str, Any]] = []
    for obj_id, rows in grouped.items():
        split = "train" if obj_id in train_ids else "validation" if obj_id in val_ids else "test"
        for row in rows:
            split_rows.append({"take_id": row.get("take_id"), "physical_object_id": obj_id, "split": split})

    leakage = defaultdict(set)
    for row in split_rows:
        leakage[str(row["physical_object_id"])].add(str(row["split"]))
    bad_groups = [obj for obj, splits in leakage.items() if len(splits) > 1]

    class_dist: dict[str, int] = defaultdict(int)
    for row in filtered:
        class_dist[str(row.get("normalized_class") or "UNKNOWN")] += 1

    obj_dist = {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)}

    (out_dir / "ml_set.json").write_text(json.dumps({
        "id": ml_set_id,
        "dataset_id": dataset_id,
        "immutable": True,
        "manifest_count": len(filtered),
        "policy": policy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "unlabeled_pool_count": len(unlabeled_pool),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# ML Set\n\nDeterministic materialization from canonical manifest.\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(out_dir),
        "validation_report": validation_report,
    }
