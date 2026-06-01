from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _group_key(row: dict[str, Any], strategy: str) -> str:
    if strategy == "by_session":
        return str(row.get("session_id") or "")
    if strategy == "by_dataset":
        return str(row.get("dataset_id") or "")
    if strategy == "chronological":
        return str(row.get("take_timestamp") or row.get("take_id") or "")
    return str(row.get("take_id") or "")


def _near_duplicate_pairs(rows: list[dict[str, Any]], feature_columns: list[str], limit: int = 20) -> list[dict[str, Any]]:
    seen: dict[str, tuple[str, int]] = {}
    pairs: list[dict[str, Any]] = []
    def _num(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0
    for row in rows:
        rounded = [f"{_num(row.get(col)):.3f}" for col in feature_columns]
        sig = hashlib.sha1("|".join(rounded).encode("utf-8")).hexdigest()
        key = str(row.get("take_id") or "")
        if sig in seen and len(pairs) < limit:
            prev_take, prev_obj = seen[sig]
            pairs.append({"take_id_a": prev_take, "object_id_a": prev_obj, "take_id_b": key, "object_id_b": int(row.get("object_id") or 0), "signature": sig})
        else:
            seen[sig] = (key, int(row.get("object_id") or 0))
    return pairs


def build_split(rows: list[dict[str, Any]], *, strategy: str, seed: int, output_dir: Path) -> dict[str, Any]:
    normalized = strategy.replace("-", "_")
    rng = random.Random(seed)
    pool = list(rows)
    if normalized == "chronological":
        pool.sort(key=lambda r: str(r.get("take_timestamp") or r.get("take_id") or ""))
    elif normalized in {"by_session", "by_dataset"}:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            grouped[_group_key(row, normalized)].append(row)
        keys = list(grouped.keys())
        rng.shuffle(keys)
        pool = [item for key in keys for item in grouped[key]]
    elif normalized in {"random", "stratified"}:
        rng.shuffle(pool)

    n = len(pool)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = pool[:n_train]
    val = pool[n_train : n_train + n_val]
    test = pool[n_train + n_val :]

    ids = {
        "train": [{"take_id": r.get("take_id"), "object_id": r.get("object_id"), "session_id": r.get("session_id"), "dataset_id": r.get("dataset_id")} for r in train],
        "validation": [{"take_id": r.get("take_id"), "object_id": r.get("object_id"), "session_id": r.get("session_id"), "dataset_id": r.get("dataset_id")} for r in val],
        "test": [{"take_id": r.get("take_id"), "object_id": r.get("object_id"), "session_id": r.get("session_id"), "dataset_id": r.get("dataset_id")} for r in test],
    }

    leakage_checks: dict[str, Any] = {}
    for field in ("take_id", "session_id", "dataset_id"):
        sets = {k: {str(item.get(field) or "") for item in v if str(item.get(field) or "")} for k, v in ids.items()}
        leakage_checks[f"duplicate_{field}_across_splits"] = {
            "train_validation": sorted(list(sets["train"].intersection(sets["validation"]))),
            "train_test": sorted(list(sets["train"].intersection(sets["test"]))),
            "validation_test": sorted(list(sets["validation"].intersection(sets["test"]))),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "strategy": normalized,
        "seed": seed,
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "train_ids": ids["train"],
        "validation_ids": ids["validation"],
        "test_ids": ids["test"],
        "leakage_checks": leakage_checks,
        "fold_metadata": {"kfold": None},
        "provenance": {"generated_by": "vision_3d_acquisition.ml.splits"},
    }
    split_path = output_dir / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    all_rows = train + val + test
    feature_cols = [k for k in rows[0].keys() if k.startswith("feature_") or k.startswith("height_") or k.startswith("surface_") or k.startswith("damage_")] if rows else []
    leakage_report = {
        "strategy": normalized,
        "seed": seed,
        "duplicate_take_ids": leakage_checks["duplicate_take_id_across_splits"],
        "duplicate_sessions": leakage_checks["duplicate_session_id_across_splits"],
        "duplicate_datasets": leakage_checks["duplicate_dataset_id_across_splits"],
        "timestamp_overlap": {"status": "not_evaluated"},
        "calibration_overlap": {"status": "not_evaluated"},
        "near_duplicate_feature_vectors": _near_duplicate_pairs(all_rows, feature_cols),
        "warnings": [],
    }
    if normalized == "by_session" and leakage_report["duplicate_sessions"]["train_test"]:
        leakage_report["warnings"].append("session_leakage_detected")
    leak_path = output_dir / "leakage_report.json"
    leak_path.write_text(json.dumps(leakage_report, indent=2) + "\n", encoding="utf-8")

    return {
        "train": train,
        "validation": val,
        "test": test,
        "split_manifest_path": str(split_path),
        "leakage_report_path": str(leak_path),
        "split_manifest": split_manifest,
        "leakage_report": leakage_report,
    }
