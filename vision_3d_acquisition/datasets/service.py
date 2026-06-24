from __future__ import annotations

import json
import math
import random
import shutil
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ML_TASK_TYPES = {"classification", "regression", "detection", "segmentation", "clustering", "benchmark"}
ML_SPLITS = {"train", "validation", "test", "holdout", "calibration", "unassigned"}
SESSION_TYPES = {"engineering", "curated", "benchmark", "operational"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in base:
        base = base.replace("--", "-")
    return base or f"id-{int(datetime.now(UTC).timestamp())}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class DatasetService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.datasets_dir = data_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for folder in self.datasets_dir.glob("dataset_*"):
            if not folder.is_dir():
                continue
            payload = _read_json(folder / "dataset.json")
            if payload:
                items.append(payload)
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        return _read_json(self._dataset_dir(dataset_id) / "dataset.json")

    def create_dataset(self, *, name: str, description: str | None = None, tags: list[str] | None = None, notes: str | None = None, dataset_id: str | None = None) -> dict[str, Any]:
        did = dataset_id or _slug(name)
        payload = {
            "id": did,
            "name": name,
            "description": description,
            "created_at": _now_iso(),
            "tags": tags or [],
            "notes": notes,
        }
        _write_json(self._dataset_dir(did) / "dataset.json", payload)
        return payload

    def update_dataset(self, dataset_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get_dataset(dataset_id)
        if not current:
            return None
        merged = {**current, **{k: v for k, v in updates.items() if v is not None or k == "notes"}}
        _write_json(self._dataset_dir(dataset_id) / "dataset.json", merged)
        return merged

    def list_sessions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        datasets = [dataset_id] if dataset_id else [item.get("id") for item in self.list_datasets()]
        for did in datasets:
            if not isinstance(did, str) or not did:
                continue
            sessions_dir = self._dataset_dir(did) / "sessions"
            if not sessions_dir.is_dir():
                continue
            for folder in sessions_dir.glob("session_*"):
                payload = _read_json(folder / "session.json")
                if payload:
                    sessions.append(payload)
        return sorted(sessions, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def get_session(self, dataset_id: str, session_id: str) -> dict[str, Any] | None:
        return _read_json(self._session_dir(dataset_id, session_id) / "session.json")

    def create_session(
        self,
        *,
        dataset_id: str,
        name: str,
        description: str | None = None,
        calibration_id: str | None = None,
        sensor_metadata: dict[str, Any] | None = None,
        conveyor_metadata: dict[str, Any] | None = None,
        lighting_metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        session_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        sid = session_id or _slug(name)
        resolved_session_type = session_type if session_type in SESSION_TYPES else "engineering"
        session_metadata = metadata if isinstance(metadata, dict) else {}
        payload = {
            "id": sid,
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "calibration_id": calibration_id,
            "sensor_metadata": sensor_metadata or {},
            "conveyor_metadata": conveyor_metadata or {},
            "lighting_metadata": lighting_metadata or {},
            "environment_metadata": {},
            "created_at": _now_iso(),
            "tags": tags or [],
            "notes": notes,
            "session_type": resolved_session_type,
            "metadata": session_metadata,
        }
        _write_json(self._session_dir(dataset_id, sid) / "session.json", payload)
        return payload

    def create_ml_set(
        self,
        *,
        dataset_id: str,
        ml_set_id: str,
        name: str,
        task_type: str,
        description: str | None = None,
        notes: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if self.get_dataset(dataset_id) is None:
            raise ValueError(f"dataset does not exist: {dataset_id}")
        if task_type not in ML_TASK_TYPES:
            raise ValueError(f"invalid task_type: {task_type}")
        now = _now_iso()
        current = self.get_ml_set(dataset_id, ml_set_id)
        if current and not overwrite:
            raise ValueError(f"ml set already exists: {dataset_id}/{ml_set_id}")
        payload = {
            "id": ml_set_id,
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "task_type": task_type,
            "created_at": str(current.get("created_at")) if current else now,
            "updated_at": now,
            "notes": notes,
        }
        _write_json(self._ml_set_dir(dataset_id, ml_set_id) / "ml_set.json", payload)
        if not (self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json").is_file():
            _write_json(self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json", {"memberships": []})
        return payload

    def get_ml_set(self, dataset_id: str, ml_set_id: str) -> dict[str, Any] | None:
        return _read_json(self._ml_set_dir(dataset_id, ml_set_id) / "ml_set.json")

    def resolve_ml_set(self, *, ml_set_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        if dataset_id:
            payload = self.get_ml_set(dataset_id, ml_set_id)
            if payload is None:
                raise ValueError(f"MLSet '{ml_set_id}' not found.")
            return payload

        matches = [item for item in self.list_ml_sets() if str(item.get("id") or "") == ml_set_id]
        if not matches:
            raise ValueError(f"MLSet '{ml_set_id}' not found.")

        dataset_ids = sorted({str(item.get("dataset_id") or "") for item in matches if str(item.get("dataset_id") or "")})
        if len(dataset_ids) > 1:
            raise ValueError(
                f"Multiple MLSets named '{ml_set_id}' exist.\n\nPlease specify:\n  --dataset-id"
            )
        return matches[0]

    def list_ml_sets(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        datasets = [dataset_id] if dataset_id else [str(item.get("id") or "") for item in self.list_datasets()]
        for did in datasets:
            if not did:
                continue
            ml_sets_dir = self._dataset_dir(did) / "ml_sets"
            if not ml_sets_dir.is_dir():
                continue
            for folder in ml_sets_dir.glob("ml_set_*"):
                if not folder.is_dir():
                    continue
                payload = _read_json(folder / "ml_set.json")
                if payload:
                    memberships = self.list_ml_set_memberships(dataset_id=did, ml_set_id=str(payload.get("id") or "")) if str(payload.get("id") or "") else []
                    items.append({
                        **payload,
                        **self.summarize_ml_set_memberships(payload, memberships),
                    })
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    @staticmethod
    def summarize_ml_set_memberships(ml_set: dict[str, Any], memberships: list[dict[str, Any]]) -> dict[str, Any]:
        membership = (ml_set.get("membership") or {}) if isinstance(ml_set, dict) else {}
        semantics = (ml_set.get("semantics") or {}) if isinstance(ml_set, dict) else {}
        explicit_mode = str(membership.get("mode") or "").strip().lower()
        if explicit_mode == "filter":
            membership_mode = "filter_snapshot"
        else:
            physical_ids = [str(row.get("physical_object_id") or "").strip() for row in memberships]
            physical_non_empty = [value for value in physical_ids if value]
            if memberships and len(physical_non_empty) == len(memberships):
                membership_mode = "physical_object_id"
            elif physical_non_empty:
                membership_mode = "mixed"
            else:
                membership_mode = explicit_mode or "ids"

        member_count = len(memberships)
        kept_rows = [row for row in memberships if bool(row.get("include", True))]
        default_trainable_rows = [row for row in memberships if bool(row.get("default_trainable", False))]
        review_required_rows = [row for row in memberships if bool(row.get("review_required", False))]
        physical_object_ids = {str(row.get("physical_object_id") or "").strip() for row in memberships if str(row.get("physical_object_id") or "").strip()}
        default_trainable_objects = {
            str(row.get("physical_object_id") or "").strip()
            for row in memberships
            if bool(row.get("default_trainable", False)) and str(row.get("physical_object_id") or "").strip()
        }
        review_required_objects = {
            str(row.get("physical_object_id") or "").strip()
            for row in memberships
            if bool(row.get("review_required", False)) and str(row.get("physical_object_id") or "").strip()
        }
        trainable_splits = {str(row.get("split") or "unassigned") for row in memberships if bool(row.get("trainable", row.get("default_trainable", False)))}
        split_status = "assigned" if trainable_splits and trainable_splits != {"unassigned"} else "unassigned"
        return {
            "member_count": member_count,
            "membership_count": member_count,
            "membership_mode": membership_mode,
            "physical_object_count": len(physical_object_ids),
            "default_trainable_count": len(default_trainable_rows),
            "default_trainable_rows": len(default_trainable_rows),
            "default_trainable_objects": len(default_trainable_objects),
            "review_required_count": len(review_required_rows),
            "review_required_rows": len(review_required_rows),
            "review_required_objects": len(review_required_objects),
            "split_status": split_status,
            "source_type": semantics.get("source_type") or (ml_set.get("provenance") or {}).get("source_type"),
            "source_filename": semantics.get("source_filename") or (ml_set.get("provenance") or {}).get("filename"),
            "source_sha256": semantics.get("source_sha256") or (ml_set.get("provenance") or {}).get("sha256"),
            "normalization_version": semantics.get("normalization_version") or ml_set.get("normalization_version"),
            "kept_in_ml_set_count": len(kept_rows),
        }

    def list_ml_set_memberships(self, *, dataset_id: str, ml_set_id: str) -> list[dict[str, Any]]:
        if self.get_ml_set(dataset_id, ml_set_id) is None:
            raise ValueError(f"unknown ml set: {dataset_id}/{ml_set_id}")
        payload = _read_json(self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json") or {}
        rows = payload.get("memberships")
        if not isinstance(rows, list):
            return []
        memberships = [item for item in rows if isinstance(item, dict)]
        return sorted(memberships, key=lambda item: str(item.get("take_id") or ""))

    def add_take_to_ml_set(
        self,
        *,
        dataset_id: str,
        ml_set_id: str,
        take_id: str,
        split: str = "unassigned",
        physical_object_id: str | None = None,
        include: bool = True,
        default_trainable: bool | None = None,
        trainable: bool | None = None,
        notes: str | None = None,
        expected_label: str | None = None,
        expected_class: str | None = None,
        expected_subclass: str | None = None,
        measurements_mm: dict[str, float] | None = None,
        raw_label: str | None = None,
        label_policy: str | None = None,
        review_required: bool | None = None,
        normalization_version: str | None = None,
        source_row: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if split not in ML_SPLITS:
            raise ValueError(f"invalid split: {split}")
        if self.get_ml_set(dataset_id, ml_set_id) is None:
            raise ValueError(f"unknown ml set: {dataset_id}/{ml_set_id}")
        take_dir = self.data_dir / "incoming" / take_id
        if not take_dir.is_dir():
            raise ValueError(f"take not found: {take_id}")
        source_metadata = _read_json(take_dir / "metadata.json") or {}
        take_meta = self.load_take_metadata(take_id=take_id, source_metadata=source_metadata)
        take_dataset_id = str(take_meta.get("dataset_id") or "")
        if not take_dataset_id:
            raise ValueError(f"take has no dataset_id: {take_id}")
        if take_dataset_id != dataset_id:
            raise ValueError(f"take dataset mismatch: take={take_id} dataset={take_dataset_id} expected={dataset_id}")

        memberships = self.list_ml_set_memberships(dataset_id=dataset_id, ml_set_id=ml_set_id)
        now = _now_iso()
        updated = False
        result_row: dict[str, Any] | None = None
        next_rows: list[dict[str, Any]] = []
        for item in memberships:
            if str(item.get("take_id") or "") != take_id:
                next_rows.append(item)
                continue
            merged = {
                **item,
                "ml_set_id": ml_set_id,
                "take_id": take_id,
                "split": split,
                "physical_object_id": physical_object_id,
                "include": bool(include),
                "default_trainable": default_trainable,
                "trainable": trainable,
                "notes": notes,
                "expected_label": expected_label,
                "expected_class": expected_class,
                "expected_subclass": expected_subclass,
                "measurements_mm": measurements_mm or {},
                "raw_label": raw_label,
                "label_policy": label_policy,
                "review_required": review_required,
                "normalization_version": normalization_version,
                "source_row": source_row,
                "extra_fields": extra_fields or {},
                "updated_at": now,
            }
            if item.get("created_at"):
                merged["created_at"] = item.get("created_at")
            else:
                merged["created_at"] = now
            result_row = merged
            next_rows.append(merged)
            updated = True
        if not updated:
            result_row = {
                "ml_set_id": ml_set_id,
                "take_id": take_id,
                "split": split,
                "physical_object_id": physical_object_id,
                "include": bool(include),
                "default_trainable": default_trainable,
                "trainable": trainable,
                "notes": notes,
                "expected_label": expected_label,
                "expected_class": expected_class,
                "expected_subclass": expected_subclass,
                "measurements_mm": measurements_mm or {},
                "raw_label": raw_label,
                "label_policy": label_policy,
                "review_required": review_required,
                "normalization_version": normalization_version,
                "source_row": source_row,
                "extra_fields": extra_fields or {},
                "created_at": now,
                "updated_at": now,
            }
            next_rows.append(result_row)

        _write_json(
            self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json",
            {"memberships": sorted(next_rows, key=lambda item: str(item.get("take_id") or ""))},
        )
        return {"status": "updated" if updated else "added", "membership": result_row}

    def import_ml_manifest(
        self,
        *,
        ml_set_id: str,
        manifest_path: Path,
        dataset_id: str | None = None,
        dry_run: bool = True,
        allow_overwrite_object_id: bool = False,
    ) -> dict[str, Any]:
        ml_set = self.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        if not manifest_path.is_file():
            raise ValueError(f"manifest not found: {manifest_path}")

        rows = _load_manifest_rows(manifest_path)
        if not rows:
            raise ValueError("manifest has no rows")

        current_memberships = self.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=ml_set_id)
        current_by_take = {str(item.get("take_id") or ""): item for item in current_memberships if str(item.get("take_id") or "")}

        label_counts: dict[str, int] = {}
        objects: set[str] = set()
        take_seen_in_manifest: dict[str, str] = {}
        missing_takes: list[str] = []
        dataset_mismatch_takes: list[str] = []
        manifest_conflicts: list[dict[str, str]] = []
        actions: list[dict[str, Any]] = []
        referenced_take_ids: set[str] = set()

        for row_index, row in enumerate(rows, start=1):
            obj_id = str(row.get("physical_object_id") or "").strip()
            if not obj_id:
                raise ValueError(f"row {row_index}: missing physical_object_id")
            take_ids = _parse_take_ids_field(row.get("take_ids"))
            if not take_ids:
                raise ValueError(f"row {row_index}: missing take_ids")
            objects.add(obj_id)

            expected_label = _clean_optional(row.get("expected_label")) or _clean_optional(row.get("label"))
            expected_subclass = _clean_optional(row.get("expected_subclass")) or _clean_optional(row.get("subclass"))
            expected_class = _clean_optional(row.get("expected_class")) or _clean_optional(row.get("class"))
            notes = _clean_optional(row.get("notes"))
            split = _clean_optional(row.get("split")) or "unassigned"
            include = _to_bool(row.get("include"), default=True)
            measurements = _extract_measurements_mm(row)
            if split not in ML_SPLITS:
                raise ValueError(f"row {row_index}: invalid split '{split}'")
            if expected_label:
                label_counts[expected_label] = label_counts.get(expected_label, 0) + 1

            for take_id in take_ids:
                referenced_take_ids.add(take_id)
                existing_obj = take_seen_in_manifest.get(take_id)
                if existing_obj and existing_obj != obj_id:
                    manifest_conflicts.append({"take_id": take_id, "object_a": existing_obj, "object_b": obj_id})
                else:
                    take_seen_in_manifest[take_id] = obj_id

                incoming_dir = self.data_dir / "incoming" / take_id
                if not incoming_dir.is_dir():
                    missing_takes.append(take_id)
                    continue
                source_metadata = _read_json(incoming_dir / "metadata.json") or {}
                take_meta = self.load_take_metadata(take_id=take_id, source_metadata=source_metadata)
                take_dataset = str(take_meta.get("dataset_id") or "")
                if take_dataset != resolved_dataset_id:
                    dataset_mismatch_takes.append(take_id)
                    continue

                current = current_by_take.get(take_id)
                if current is not None:
                    current_object_id = str(current.get("physical_object_id") or "").strip()
                    if current_object_id and current_object_id != obj_id and not allow_overwrite_object_id:
                        manifest_conflicts.append({"take_id": take_id, "object_a": current_object_id, "object_b": obj_id})

                actions.append(
                    {
                        "take_id": take_id,
                        "physical_object_id": obj_id,
                        "split": split,
                        "include": include,
                        "notes": notes,
                        "expected_label": expected_label,
                        "expected_class": expected_class,
                        "expected_subclass": expected_subclass,
                        "measurements_mm": measurements,
                    }
                )

        if manifest_conflicts:
            preview = ", ".join(sorted({item["take_id"] for item in manifest_conflicts})[:20])
            raise ValueError(f"conflicting physical_object_id assignments for takes: {preview}")
        if missing_takes:
            preview = ", ".join(sorted(set(missing_takes))[:20])
            raise ValueError(f"missing takes: {preview}")
        if dataset_mismatch_takes:
            preview = ", ".join(sorted(set(dataset_mismatch_takes))[:20])
            raise ValueError(f"dataset mismatch for takes: {preview}")

        added = 0
        updated = 0
        if not dry_run:
            for action in actions:
                result = self.add_take_to_ml_set(
                    dataset_id=resolved_dataset_id,
                    ml_set_id=ml_set_id,
                    take_id=action["take_id"],
                    split=action["split"],
                    physical_object_id=action["physical_object_id"],
                    include=bool(action["include"]),
                    notes=action["notes"],
                    expected_label=action["expected_label"],
                    expected_class=action["expected_class"],
                    expected_subclass=action["expected_subclass"],
                    measurements_mm=action["measurements_mm"],
                )
                if result["status"] == "added":
                    added += 1
                else:
                    updated += 1
        else:
            for action in actions:
                if action["take_id"] in current_by_take:
                    updated += 1
                else:
                    added += 1

        return {
            "manifest_path": str(manifest_path),
            "ml_set_id": ml_set_id,
            "dataset_id": resolved_dataset_id,
            "dry_run": bool(dry_run),
            "rows_total": len(rows),
            "takes_total_referenced": len(referenced_take_ids),
            "takes_valid": len(referenced_take_ids),
            "takes_missing": 0,
            "membership_added": added,
            "membership_updated": updated,
            "physical_objects_total": len(objects),
            "labels_summary": dict(sorted(label_counts.items(), key=lambda item: item[0])),
            "conflicts": manifest_conflicts,
        }

    def remove_take_from_ml_set(self, *, dataset_id: str, ml_set_id: str, take_id: str) -> bool:
        memberships = self.list_ml_set_memberships(dataset_id=dataset_id, ml_set_id=ml_set_id)
        next_rows = [item for item in memberships if str(item.get("take_id") or "") != take_id]
        if len(next_rows) == len(memberships):
            return False
        _write_json(self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json", {"memberships": next_rows})
        return True

    def assign_ml_set_splits(
        self,
        *,
        ml_set_id: str,
        dataset_id: str | None = None,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        holdout_ratio: float = 0.0,
        calibration_ratio: float = 0.0,
        by_physical_object_id: bool = True,
        seed: int = 42,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        ratios = {
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
            "holdout": float(holdout_ratio),
            "calibration": float(calibration_ratio),
        }
        for name, value in ratios.items():
            if value < 0:
                raise ValueError(f"invalid ratio for {name}: must be >= 0")
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("invalid ratios: train+validation+test+holdout+calibration must sum to 1.0")

        ml_set = self.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        memberships = self.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=ml_set_id)
        if not memberships:
            raise ValueError("ml set has no memberships")

        active = [
            row
            for row in memberships
            if bool(row.get("trainable", row.get("default_trainable", row.get("include", True))))
        ]
        if not active:
            raise ValueError("ml set has no trainable memberships")

        mode = "physical_object_id" if by_physical_object_id else "take_id"
        missing_physical_object_takes: list[str] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in active:
            take_id = str(row.get("take_id") or "")
            if not take_id:
                continue
            if by_physical_object_id:
                key = str(row.get("physical_object_id") or "").strip()
                if not key:
                    missing_physical_object_takes.append(take_id)
                    continue
            else:
                key = take_id
            grouped.setdefault(key, []).append(row)

        if missing_physical_object_takes:
            preview = ", ".join(sorted(missing_physical_object_takes)[:20])
            raise ValueError(f"missing physical_object_id for takes: {preview}")

        group_rows: list[dict[str, Any]] = []
        for key, rows in grouped.items():
            take_ids = sorted(str(item.get("take_id") or "") for item in rows)
            group_rows.append({"group_key": key, "take_ids": take_ids, "take_count": len(take_ids)})
        group_rows.sort(key=lambda item: str(item["group_key"]))

        rng = random.Random(int(seed))
        shuffled = list(group_rows)
        rng.shuffle(shuffled)

        split_order = [name for name in ["train", "validation", "test", "holdout", "calibration"] if ratios[name] > 0]
        total_groups = len(shuffled)
        remaining_groups = total_groups
        remaining_ratio = sum(ratios[name] for name in split_order)
        target_groups: dict[str, int] = {}
        assigned_count = 0
        for index, split in enumerate(split_order):
            if index == len(split_order) - 1:
                count = remaining_groups
            else:
                expected = (ratios[split] / remaining_ratio) * remaining_groups if remaining_ratio > 0 else 0.0
                count = int(round(expected))
                count = max(0, min(count, remaining_groups))
            target_groups[split] = count
            assigned_count += count
            remaining_groups -= count
            remaining_ratio -= ratios[split]
        if assigned_count < total_groups and split_order:
            target_groups[split_order[-1]] += total_groups - assigned_count

        assigned_by_group: dict[str, str] = {}
        cursor = 0
        for split in split_order:
            count = target_groups.get(split, 0)
            for row in shuffled[cursor : cursor + count]:
                assigned_by_group[str(row["group_key"])] = split
            cursor += count
        for row in shuffled[cursor:]:
            assigned_by_group[str(row["group_key"])] = split_order[-1]

        split_counts = {split: {"groups": 0, "takes": 0} for split in split_order}
        assignment_rows: list[dict[str, Any]] = []
        for group in group_rows:
            split = assigned_by_group[str(group["group_key"])]
            split_counts[split]["groups"] += 1
            split_counts[split]["takes"] += int(group["take_count"])
            assignment_rows.append({"group_key": group["group_key"], "split": split, "take_ids": group["take_ids"]})

        if not dry_run:
            now = _now_iso()
            split_by_take: dict[str, str] = {}
            for item in assignment_rows:
                for take_id in item["take_ids"]:
                    split_by_take[str(take_id)] = str(item["split"])
            next_rows: list[dict[str, Any]] = []
            for row in memberships:
                copied = dict(row)
                take_id = str(copied.get("take_id") or "")
                if bool(copied.get("trainable", copied.get("default_trainable", copied.get("include", True)))) and take_id in split_by_take:
                    copied["split"] = split_by_take[take_id]
                    copied["updated_at"] = now
                next_rows.append(copied)
            _write_json(
                self._ml_set_dir(resolved_dataset_id, ml_set_id) / "memberships.json",
                {"memberships": sorted(next_rows, key=lambda item: str(item.get("take_id") or ""))},
            )

        return {
            "ml_set_id": ml_set_id,
            "dataset_id": resolved_dataset_id,
            "dry_run": bool(dry_run),
            "seed": int(seed),
            "assignment_mode": mode,
            "ratios": ratios,
            "group_count": len(group_rows),
            "take_count": len(active),
            "split_counts": split_counts,
            "assignments": assignment_rows,
        }

    def reprocess_ml_set(
        self,
        *,
        ml_set_id: str,
        pipeline_id: str,
        dataset_id: str | None = None,
        splits: list[str] | None = None,
        include_only: bool = True,
        reprocess: bool = True,
        dry_run: bool = True,
        classifier_rules_path: str | None = None,
    ) -> dict[str, Any]:
        from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing  # noqa: WPS433

        ml_set = self.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        memberships = self.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=ml_set_id)
        if include_only:
            memberships = [row for row in memberships if bool(row.get("include", True))]

        allowed_splits = {str(item) for item in (splits or []) if str(item)}
        if allowed_splits:
            memberships = [row for row in memberships if str(row.get("split") or "unassigned") in allowed_splits]

        selected_take_ids = sorted({str(row.get("take_id") or "") for row in memberships if str(row.get("take_id") or "")})
        if not selected_take_ids:
            raise ValueError("no memberships selected for reprocessing")

        result: dict[str, Any] = {
            "ml_set_id": ml_set_id,
            "dataset_id": resolved_dataset_id,
            "pipeline_id": pipeline_id,
            "selected_take_ids": selected_take_ids,
            "selected_count": len(selected_take_ids),
            "split_counts": self._split_counts_from_memberships(memberships),
            "processed": [],
            "failures": [],
            "success_count": 0,
            "failure_count": 0,
            "dry_run": bool(dry_run),
            "classifier_rules_path": classifier_rules_path,
        }
        if dry_run:
            return result

        settings = self._api_settings()
        for take_id in selected_take_ids:
            try:
                dispatch = dispatch_take_processing(
                    settings=settings,
                    take_id=take_id,
                    pipeline_id=pipeline_id,
                    reprocess=bool(reprocess),
                    source_id=None,
                    recipe_version_id=None,
                    acquisition_group_id=None,
                    calibration_profile_id=None,
                    stage_params=(
                        {"classify_25d": {"classifier_rules_path": classifier_rules_path}}
                        if classifier_rules_path
                        else None
                    ),
                )
                result["processed"].append({"take_id": take_id, "status": str(dispatch.get("status") or "ok")})
                result["success_count"] += 1
            except Exception as exc:
                result["failures"].append({"take_id": take_id, "error": str(exc)})
                result["failure_count"] += 1
        return result

    def export_ml_set_features(
        self,
        *,
        ml_set_id: str,
        pipeline_id: str,
        output_path: Path,
        dataset_id: str | None = None,
        splits: list[str] | None = None,
        require_processed: bool = False,
        include_diagnostics: bool = False,
        include_invalidity_flags: bool = False,
        include_provenance_summary: bool = False,
    ) -> dict[str, Any]:
        ml_set = self.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        memberships = self.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=ml_set_id)
        memberships = [row for row in memberships if bool(row.get("include", True))]
        allowed_splits = {str(item) for item in (splits or []) if str(item)}
        if allowed_splits:
            memberships = [row for row in memberships if str(row.get("split") or "unassigned") in allowed_splits]
        if not memberships:
            raise ValueError("no memberships selected for export")

        from vision_3d_acquisition.api.filesystem import get_take_detail  # noqa: WPS433

        settings = self._api_settings()
        rows: list[dict[str, Any]] = []
        missing_processed: list[str] = []
        for membership in sorted(memberships, key=lambda item: str(item.get("take_id") or "")):
            take_id = str(membership.get("take_id") or "")
            detail = get_take_detail(settings, take_id)
            if detail is None:
                missing_processed.append(take_id)
                continue
            take_metadata = detail.take_metadata if isinstance(detail.take_metadata, dict) else {}
            result = detail.result if isinstance(detail.result, dict) else None
            if not self._is_result_compatible_with_pipeline(result, pipeline_id):
                missing_processed.append(take_id)
                if require_processed:
                    continue
                result = None

            row: dict[str, Any] = {
                "take_id": take_id,
                "dataset_id": resolved_dataset_id,
                "session_id": str((take_metadata.get("session_id") if isinstance(take_metadata, dict) else None) or ""),
                "ml_set_id": ml_set_id,
                "split": str(membership.get("split") or "unassigned"),
                "physical_object_id": str(membership.get("physical_object_id") or ""),
                "expected_class": take_metadata.get("expected_class"),
                "expected_subclass": take_metadata.get("expected_subclass"),
                "expected_label": take_metadata.get("expected_label"),
                "pipeline_id": pipeline_id if result is None else str(((result.get("processing_pipeline") or {}).get("id") or pipeline_id)),
                "pipeline_run_id": None if result is None else result.get("run_id"),
                "pipeline_timestamp": None if result is None else (result.get("processed_at") or result.get("timestamp")),
                "classifier_engine": None if result is None else ((result.get("classification") or {}).get("classifier_engine") if isinstance(result.get("classification"), dict) else None),
                "rule_set_id": None if result is None else ((result.get("classification") or {}).get("rule_set_id") if isinstance(result.get("classification"), dict) else None),
                "rule_set_version": None if result is None else ((result.get("classification") or {}).get("rule_set_version") if isinstance(result.get("classification"), dict) else None),
                "rule_set_source": None if result is None else ((result.get("classification") or {}).get("rule_set_source") if isinstance(result.get("classification"), dict) else None),
            }
            if result is not None:
                feature_values = _extract_scalar_features(result)
                for key, value in feature_values.items():
                    row[key] = value
                if include_diagnostics:
                    diagnostics = _read_json(self.data_dir / "processed" / take_id / "feature_vector.json")
                    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("features"), dict):
                        for key, value in diagnostics["features"].items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                row[f"diag_{key}"] = value
                if include_invalidity_flags:
                    flags = _read_json(self.data_dir / "processed" / take_id / "quality_flags.json")
                    codes: list[str] = []
                    if isinstance(flags, dict) and isinstance(flags.get("flags"), list):
                        for item in flags["flags"]:
                            if isinstance(item, dict):
                                code = str(item.get("code") or "").strip()
                                if code:
                                    codes.append(code)
                    row["diagnostic_flag_count"] = len(codes)
                    row["diagnostic_flags"] = "|".join(sorted(set(codes)))
                if include_provenance_summary:
                    provenance = _read_json(self.data_dir / "processed" / take_id / "feature_provenance.json")
                    if isinstance(provenance, dict):
                        source_stages = sorted(
                            {
                                str(item.get("source_stage") or "")
                                for item in provenance.values()
                                if isinstance(item, dict) and str(item.get("source_stage") or "")
                            }
                        )
                        warning_count = sum(
                            1
                            for item in provenance.values()
                            if isinstance(item, dict) and str(item.get("validity") or "") in {"warning", "invalid"}
                        )
                        row["provenance_source_stage_count"] = len(source_stages)
                        row["provenance_source_stages"] = "|".join(source_stages)
                        row["provenance_warning_or_invalid_count"] = warning_count
            rows.append(row)

        if require_processed and missing_processed:
            raise ValueError(f"missing processing output for pipeline '{pipeline_id}': {', '.join(sorted(missing_processed)[:20])}")
        if not rows:
            raise ValueError("no feature rows to export")

        base_columns = [
            "take_id",
            "dataset_id",
            "session_id",
            "ml_set_id",
            "split",
            "physical_object_id",
            "expected_class",
            "expected_subclass",
            "expected_label",
            "pipeline_id",
            "pipeline_run_id",
            "pipeline_timestamp",
            "classifier_engine",
            "rule_set_id",
            "rule_set_version",
            "rule_set_source",
        ]
        feature_columns = sorted({key for row in rows for key in row.keys() if key not in set(base_columns)})
        all_columns = base_columns + feature_columns

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col) for col in all_columns})

        return {
            "ml_set_id": ml_set_id,
            "dataset_id": resolved_dataset_id,
            "pipeline_id": pipeline_id,
            "output_path": str(output_path),
            "row_count": len(rows),
            "column_count": len(all_columns),
            "columns": all_columns,
            "missing_processed_take_ids": sorted(set(missing_processed)),
            "split_counts": self._split_counts_from_memberships(memberships),
            "include_diagnostics": bool(include_diagnostics),
            "include_invalidity_flags": bool(include_invalidity_flags),
            "include_provenance_summary": bool(include_provenance_summary),
        }

    def update_session(self, dataset_id: str, session_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get_session(dataset_id, session_id)
        if not current:
            return None
        if "session_type" in updates and updates["session_type"] not in SESSION_TYPES:
            updates["session_type"] = current.get("session_type") or "engineering"
        merged = {**current, **{k: v for k, v in updates.items() if v is not None or k == "notes"}}
        _write_json(self._session_dir(dataset_id, session_id) / "session.json", merged)
        return merged

    def resolve_take_membership(self, take_id: str) -> tuple[str | None, str | None]:
        memberships = self.resolve_all_take_memberships(take_id)
        if memberships:
            did, sid = memberships[0]
            return did, sid
        return None, None

    def resolve_all_take_memberships(self, take_id: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for dataset in self.list_datasets():
            did = str(dataset.get("id") or "")
            if not did:
                continue
            sessions_dir = self._dataset_dir(did) / "sessions"
            if not sessions_dir.is_dir():
                continue
            for session_folder in sessions_dir.glob("session_*"):
                sid = session_folder.name.replace("session_", "", 1)
                take_folder = session_folder / "takes" / take_id
                if take_folder.is_dir():
                    rows.append((did, sid))
        return rows

    def update_take_session_id(
        self,
        *,
        take_id: str,
        dataset_id: str,
        new_session_id: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memberships = self.resolve_all_take_memberships(take_id)
        mismatched = [did for did, _sid in memberships if did != dataset_id]
        if mismatched:
            raise ValueError(f"take {take_id} belongs to multiple datasets: {sorted(set(mismatched + [dataset_id]))}")

        current = self.load_take_metadata(take_id=take_id, source_metadata=source_metadata)
        current_dataset = str(current.get("dataset_id") or "")
        if current_dataset and current_dataset != dataset_id:
            raise ValueError(f"take {take_id} dataset mismatch: current={current_dataset} expected={dataset_id}")

        updated = self.upsert_take_metadata(
            take_id=take_id,
            dataset_id=dataset_id,
            session_id=new_session_id,
            updates={"session_id": new_session_id},
            source_metadata=source_metadata,
        )

        # Keep membership canonical: one take sidecar per dataset/session.
        target_take_dir = self._session_dir(dataset_id, new_session_id) / "takes" / take_id
        for did, sid in memberships:
            stale_take_dir = self._session_dir(did, sid) / "takes" / take_id
            if stale_take_dir == target_take_dir:
                continue
            if stale_take_dir.exists():
                shutil.rmtree(stale_take_dir, ignore_errors=True)

        return updated

    def batch_update_take_session(
        self,
        *,
        take_ids: list[str],
        new_session_id: str,
        dataset_id: str | None = None,
        create_session: bool = False,
        session_name: str | None = None,
        allow_missing: bool = False,
        apply: bool = False,
    ) -> dict[str, Any]:
        deduped_take_ids: list[str] = []
        seen: set[str] = set()
        for item in take_ids:
            cleaned = str(item).strip()
            if cleaned and cleaned not in seen:
                deduped_take_ids.append(cleaned)
                seen.add(cleaned)
        if not deduped_take_ids:
            raise ValueError("no takes provided")

        valid_take_ids = [take_id for take_id in deduped_take_ids if (self.data_dir / "incoming" / take_id).is_dir()]
        missing_take_ids = [take_id for take_id in deduped_take_ids if take_id not in set(valid_take_ids)]
        if missing_take_ids and not allow_missing:
            raise ValueError("missing takes found")
        if not valid_take_ids:
            raise ValueError("no valid takes to update")

        rows: list[dict[str, Any]] = []
        for take_id in valid_take_ids:
            source_path = self.data_dir / "incoming" / take_id / "metadata.json"
            source_metadata = _read_json(source_path) or {}
            take_meta = self.load_take_metadata(take_id=take_id, source_metadata=source_metadata)
            rows.append(
                {
                    "take_id": take_id,
                    "dataset_id": str(take_meta.get("dataset_id") or ""),
                    "old_session_id": str(take_meta.get("session_id") or source_metadata.get("session_id") or ""),
                    "source_metadata": source_metadata,
                }
            )

        dataset_ids = sorted({str(row.get("dataset_id") or "") for row in rows if str(row.get("dataset_id") or "")})
        if len(dataset_ids) > 1 and not dataset_id:
            raise ValueError(f"takes belong to multiple datasets: {', '.join(dataset_ids)}")
        resolved_dataset_id = str(dataset_id or (dataset_ids[0] if dataset_ids else ""))
        if not resolved_dataset_id:
            raise ValueError("could not infer dataset_id from takes")
        if dataset_id:
            mismatched = [str(row.get("take_id") or "") for row in rows if row.get("dataset_id") and row.get("dataset_id") != resolved_dataset_id]
            if mismatched:
                raise ValueError(f"--dataset-id={resolved_dataset_id} does not match dataset of takes: {', '.join(mismatched[:10])}")
            for row in rows:
                if not row.get("dataset_id"):
                    row["dataset_id"] = resolved_dataset_id

        destination = self.get_session(resolved_dataset_id, new_session_id)
        created_session: dict[str, Any] | None = None
        if destination is None:
            if not create_session:
                raise ValueError(f"destination session does not exist: dataset={resolved_dataset_id} session={new_session_id}")
            if self.get_dataset(resolved_dataset_id) is None:
                raise ValueError(f"dataset does not exist: {resolved_dataset_id}")
            if apply:
                created_session = self.create_session(
                    dataset_id=resolved_dataset_id,
                    session_id=new_session_id,
                    name=session_name or new_session_id,
                    notes="Created by update-take-session CLI",
                )
            else:
                created_session = {
                    "id": new_session_id,
                    "dataset_id": resolved_dataset_id,
                    "name": session_name or new_session_id,
                    "created_at": _now_iso(),
                    "notes": "Created by update-take-session CLI",
                }

        updated_rows: list[dict[str, str]] = []
        if apply:
            for row in rows:
                take_id = str(row["take_id"])
                self.update_take_session_id(
                    take_id=take_id,
                    dataset_id=resolved_dataset_id,
                    new_session_id=new_session_id,
                    source_metadata=row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else None,
                )
                updated_rows.append(
                    {
                        "take_id": take_id,
                        "old_session_id": str(row.get("old_session_id") or ""),
                        "new_session_id": new_session_id,
                    }
                )

        return {
            "requested": len(deduped_take_ids),
            "valid_take_ids": valid_take_ids,
            "missing_take_ids": missing_take_ids,
            "rows": [{"take_id": str(r["take_id"]), "dataset_id": str(r["dataset_id"]), "old_session_id": str(r["old_session_id"])} for r in rows],
            "dataset_id": resolved_dataset_id,
            "destination_session_exists": destination is not None,
            "created_session": created_session,
            "updated_rows": updated_rows,
        }

    def upsert_take_metadata(
        self,
        *,
        take_id: str,
        dataset_id: str,
        session_id: str,
        updates: dict[str, Any],
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        take_path = self._session_dir(dataset_id, session_id) / "takes" / take_id / "metadata.json"
        existing = _read_json(take_path) or self.default_take_metadata(take_id, source_metadata)
        resolved_dataset = updates["dataset_id"] if "dataset_id" in updates else dataset_id
        resolved_session = updates["session_id"] if "session_id" in updates else session_id
        merged = {**existing, **updates, "take_id": take_id, "dataset_id": resolved_dataset, "session_id": resolved_session}
        _write_json(take_path, merged)
        return merged

    def load_take_metadata(self, *, take_id: str, source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        did, sid = self.resolve_take_membership(take_id)
        if did and sid:
            payload = _read_json(self._session_dir(did, sid) / "takes" / take_id / "metadata.json")
            if payload:
                payload.setdefault("dataset_id", did)
                payload.setdefault("session_id", sid)
                return payload
        fallback = self.default_take_metadata(take_id, source_metadata)
        if did:
            fallback["dataset_id"] = did
        if sid:
            fallback["session_id"] = sid
        return fallback

    def default_take_metadata(self, take_id: str, source_metadata: dict[str, Any] | None) -> dict[str, Any]:
        source = source_metadata or {}
        return {
            "take_id": take_id,
            "friendly_name": take_id,
            "labels": [],
            "tags": [],
            "semantic_labels": [],
            "superclass_labels": [],
            "normalized_class": None,
            "normalization_version": None,
            "notes": None,
            "categories": [],
            "reference_type": None,
            "is_reference": False,
            "is_golden_sample": False,
            "expected_class": None,
            "expected_diameter_mm": None,
            "expected_count": None,
            "physical_object_id": None,
            "operator_notes": None,
            "session_notes": None,
            "validation_status": "unreviewed",
            "dataset_id": None,
            "session_id": source.get("session_id"),
            "acquisition_group_id": source.get("acquisition_group_id"),
            "created_at": source.get("created_at") or _now_iso(),
            "object_annotations": [],
            "archived": False,
            "archived_at": None,
            "archived_reason": None,
        }

    def iter_dataset_takes(self, dataset_id: str, *, session_id: str | None = None) -> list[tuple[str, str, dict[str, Any]]]:
        sessions = [session_id] if session_id else [str(item.get("id") or "") for item in self.list_sessions(dataset_id=dataset_id)]
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for sid in sessions:
            if not sid:
                continue
            takes_dir = self._session_dir(dataset_id, sid) / "takes"
            if not takes_dir.is_dir():
                continue
            for child in sorted(takes_dir.iterdir()):
                if not child.is_dir():
                    continue
                take_id = child.name
                payload = _read_json(child / "metadata.json") or self.default_take_metadata(take_id, source_metadata={"session_id": sid})
                rows.append((sid, take_id, payload))
        return rows

    def visible_take_counts_by_session(self, dataset_id: str, *, include_archived: bool = False) -> dict[str, int]:
        counts: dict[str, int] = {}
        for session in self.list_sessions(dataset_id=dataset_id):
            sid = str(session.get("id") or "")
            if not sid:
                continue
            counts[sid] = 0
            takes_dir = self._session_dir(dataset_id, sid) / "takes"
            if not takes_dir.is_dir():
                continue
            for child in takes_dir.iterdir():
                if not child.is_dir():
                    continue
                payload = _read_json(child / "metadata.json") or {}
                if not include_archived and bool(payload.get("archived")):
                    continue
                counts[sid] += 1
        return counts

    def label_summary(self, dataset_id: str) -> dict[str, Any]:
        raw_counts: dict[str, int] = {}
        semantic_counts: dict[str, int] = {}
        superclass_counts: dict[str, int] = {}
        unmapped_counts: dict[str, int] = {}
        normalization_versions: dict[str, int] = {}
        take_count = 0
        for _sid, _take_id, payload in self.iter_dataset_takes(dataset_id):
            take_count += 1
            for tag in [str(item).strip() for item in (payload.get("tags") or []) if str(item).strip()]:
                raw_counts[tag] = raw_counts.get(tag, 0) + 1
            for label in [str(item).strip() for item in (payload.get("semantic_labels") or []) if str(item).strip()]:
                semantic_counts[label] = semantic_counts.get(label, 0) + 1
            for label in [str(item).strip() for item in (payload.get("superclass_labels") or []) if str(item).strip()]:
                superclass_counts[label] = superclass_counts.get(label, 0) + 1
            for warning in [str(item) for item in (payload.get("normalization_warnings") or []) if str(item)]:
                if warning.startswith("UNMAPPED_TAG:"):
                    tag = warning.split(":", 1)[1]
                    unmapped_counts[tag] = unmapped_counts.get(tag, 0) + 1
            version = str(payload.get("normalization_version") or "").strip()
            if version:
                normalization_versions[version] = normalization_versions.get(version, 0) + 1
        dominant_version = None
        if normalization_versions:
            dominant_version = sorted(normalization_versions.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return {
            "dataset_id": dataset_id,
            "take_count": take_count,
            "raw_tag_counts": dict(sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))),
            "semantic_label_counts": dict(sorted(semantic_counts.items(), key=lambda item: (-item[1], item[0]))),
            "superclass_counts": dict(sorted(superclass_counts.items(), key=lambda item: (-item[1], item[0]))),
            "unmapped_tags": dict(sorted(unmapped_counts.items(), key=lambda item: (-item[1], item[0]))),
            "normalization_version": dominant_version,
            "normalization_versions_seen": normalization_versions,
        }

    def upsert_object_annotation(
        self,
        *,
        take_id: str,
        dataset_id: str,
        session_id: str,
        annotation: dict[str, Any],
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.upsert_take_metadata(
            take_id=take_id,
            dataset_id=dataset_id,
            session_id=session_id,
            updates={},
            source_metadata=source_metadata,
        )
        current = payload.get("object_annotations")
        annotations = [item for item in current if isinstance(item, dict)] if isinstance(current, list) else []
        annotation_id = str(annotation.get("id") or "").strip() or f"object_{len(annotations) + 1:03d}"
        now = _now_iso()
        incoming = {
            "id": annotation_id,
            "source_stage": annotation.get("source_stage"),
            "source_artifact_id": annotation.get("source_artifact_id"),
            "candidate_id": str(annotation.get("candidate_id") or ""),
            "bbox": annotation.get("bbox"),
            "centroid": annotation.get("centroid"),
            "contour_ref": annotation.get("contour_ref"),
            "labels": [str(v) for v in (annotation.get("labels") or []) if str(v)],
            "expected_class": annotation.get("expected_class"),
            "expected_diameter_mm": annotation.get("expected_diameter_mm"),
            "notes": annotation.get("notes"),
            "validation_status": annotation.get("validation_status") or "unreviewed",
            "created_at": annotation.get("created_at") or now,
            "updated_at": now,
        }
        updated = False
        next_annotations: list[dict[str, Any]] = []
        for existing in annotations:
            if str(existing.get("id") or "") == annotation_id:
                merged = {**existing, **incoming}
                if existing.get("created_at"):
                    merged["created_at"] = existing.get("created_at")
                next_annotations.append(merged)
                updated = True
            else:
                next_annotations.append(existing)
        if not updated:
            next_annotations.append(incoming)
        payload["object_annotations"] = next_annotations
        take_path = self._session_dir(dataset_id, session_id) / "takes" / take_id / "metadata.json"
        _write_json(take_path, payload)
        return incoming

    def match_object_annotations(
        self,
        *,
        annotations: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not annotations:
            return []
        by_candidate_id = {str(c.get("candidate_id") or ""): c for c in candidates if str(c.get("candidate_id") or "")}
        matched: list[dict[str, Any]] = []
        for annotation in annotations:
            item = dict(annotation)
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id and candidate_id in by_candidate_id:
                item["matched_candidate_id"] = candidate_id
                item["matched_by"] = "candidate_id"
                matched.append(item)
                continue
            resolved = _match_by_geometry(item, candidates)
            item["matched_candidate_id"] = resolved.get("candidate_id") if resolved else None
            item["matched_by"] = resolved.get("matched_by") if resolved else None
            matched.append(item)
        return matched

    def _dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / f"dataset_{dataset_id}"

    def _session_dir(self, dataset_id: str, session_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "sessions" / f"session_{session_id}"

    def _ml_set_dir(self, dataset_id: str, ml_set_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "ml_sets" / f"ml_set_{ml_set_id}"

    def _api_settings(self):
        from vision_3d_acquisition.api.settings import ApiSettings  # noqa: WPS433

        settings = ApiSettings(
            data_dir=self.data_dir,
            incoming_dir=self.data_dir / "incoming",
            processed_dir=self.data_dir / "processed",
            state_dir=self.data_dir / "state",
            events_dir=self.data_dir / "events",
            sessions_dir=self.data_dir / "sessions",
            datasets_dir=self.data_dir / "datasets",
        )
        settings.ensure_directories()
        return settings

    @staticmethod
    def _is_result_compatible_with_pipeline(result: dict[str, Any] | None, pipeline_id: str) -> bool:
        if not isinstance(result, dict):
            return False
        resolved = str(((result.get("processing_pipeline") or {}).get("id") or "")).strip()
        return resolved == pipeline_id

    @staticmethod
    def _split_counts_from_memberships(memberships: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in memberships:
            split = str(row.get("split") or "unassigned")
            counts[split] = counts.get(split, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))


def _match_by_geometry(annotation: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    ann_bbox = _quad(annotation.get("bbox"))
    ann_centroid = _pair(annotation.get("centroid"))
    if ann_bbox is not None:
        best_iou = 0.0
        best: dict[str, Any] | None = None
        for candidate in candidates:
            bbox = _quad(candidate.get("bbox"))
            if bbox is None:
                continue
            iou = _bbox_iou(ann_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best = candidate
        if best is not None and best_iou >= 0.2:
            return {**best, "matched_by": "bbox_iou"}
    if ann_centroid is not None:
        nearest: dict[str, Any] | None = None
        best_dist = float("inf")
        for candidate in candidates:
            centroid = _pair(candidate.get("centroid"))
            if centroid is None:
                continue
            distance = math.dist(ann_centroid, centroid)
            if distance < best_dist:
                best_dist = distance
                nearest = candidate
        if nearest is not None:
            return {**nearest, "matched_by": "nearest_centroid"}
    return None


def _pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _quad(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    except Exception:
        return None


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = (aw * ah) + (bw * bh) - intersection
    return (intersection / union) if union > 0 else 0.0


def _extract_scalar_features(payload: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    allowed_roots = [
        ("summary", payload.get("summary")),
        ("timing_ms", payload.get("timing_ms")),
        ("classification_diagnostics", payload.get("classification_diagnostics")),
    ]
    for prefix, value in allowed_roots:
        if isinstance(value, dict):
            _flatten_numeric_dict(prefix, value, features)
    objects = payload.get("objects")
    if isinstance(objects, list) and objects:
        numeric: dict[str, list[float]] = {}
        for item in objects:
            if not isinstance(item, dict):
                continue
            flat: dict[str, Any] = {}
            _flatten_numeric_dict("object", item, flat)
            for key, value in flat.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(key, []).append(float(value))
        for key, values in numeric.items():
            if not values:
                continue
            features[f"{key}_mean"] = sum(values) / len(values)
            features[f"{key}_min"] = min(values)
            features[f"{key}_max"] = max(values)
    return dict(sorted(features.items(), key=lambda item: item[0]))


def _flatten_numeric_dict(prefix: str, value: dict[str, Any], out: dict[str, Any]) -> None:
    for key in sorted(value.keys()):
        item = value[key]
        full = f"{prefix}_{key}"
        if isinstance(item, dict):
            _flatten_numeric_dict(full, item, out)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out[full] = item


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [dict(row) for row in reader if isinstance(row, dict)]
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [item for item in payload["rows"] if isinstance(item, dict)]
        raise ValueError("json manifest must be a list of objects or {\"rows\": [...]}")
    raise ValueError("unsupported manifest format; use .csv or .json")


def _parse_take_ids_field(value: Any) -> list[str]:
    if isinstance(value, list):
        tokens = [str(item).strip() for item in value]
    else:
        raw = str(value or "")
        tokens = [item.strip() for item in raw.split(";")]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in tokens:
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _clean_optional(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _to_bool(value: Any, *, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _extract_measurements_mm(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("d1_mm", "d2_mm", "d3_mm"):
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            out[key] = float(value)
        except Exception as exc:
            raise ValueError(f"invalid numeric value for {key}: {value}") from exc
    return out
