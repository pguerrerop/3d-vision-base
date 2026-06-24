from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.filesystem import get_take_summary
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


class MLSetSummaryService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings.data_dir)

    def _resolve_ml_set(self, ml_set_id: str, dataset_id: str | None = None) -> tuple[dict[str, Any], str]:
        ml_set = self.dataset_service.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or dataset_id or "")
        if not resolved_dataset_id:
            raise ValueError(f"Unable to resolve dataset for ML set: {ml_set_id}")
        return ml_set, resolved_dataset_id

    def _memberships_with_metadata(self, ml_set_id: str, dataset_id: str | None = None) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        ml_set, resolved_dataset_id = self._resolve_ml_set(ml_set_id, dataset_id)
        memberships = self.dataset_service.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=ml_set_id)
        enriched: list[dict[str, Any]] = []
        for row in memberships:
            take_id = str(row.get("take_id") or "")
            if not take_id:
                continue
            source_metadata = _safe_read_json(self.settings.incoming_dir / take_id / "metadata.json")
            take_meta = self.dataset_service.load_take_metadata(take_id=take_id, source_metadata=source_metadata)
            summary = get_take_summary(self.settings, take_id)
            enriched.append({
                **row,
                "take_metadata": take_meta,
                "take_summary": summary.model_dump(mode="json"),
            })
        return ml_set, resolved_dataset_id, enriched

    def build_summary(self, ml_set_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        ml_set, resolved_dataset_id, memberships = self._memberships_with_metadata(ml_set_id, dataset_id)
        dataset = self.dataset_service.get_dataset(resolved_dataset_id) or {"id": resolved_dataset_id, "name": resolved_dataset_id}
        membership_overview = self.dataset_service.summarize_ml_set_memberships(ml_set, memberships)
        kept_rows = [row for row in memberships if bool(row.get("include", True))]
        excluded_rows = [row for row in memberships if not bool(row.get("include", True))]
        trainable_rows = [
            row
            for row in kept_rows
            if bool(row.get("trainable", row.get("default_trainable", False)))
        ]
        object_ids = {str(row.get("physical_object_id") or "") for row in kept_rows if str(row.get("physical_object_id") or "")}
        trainable_object_ids = {str(row.get("physical_object_id") or "") for row in trainable_rows if str(row.get("physical_object_id") or "")}
        review_required_object_ids = {
            str(row.get("physical_object_id") or "")
            for row in memberships
            if bool(row.get("review_required", False)) and str(row.get("physical_object_id") or "")
        }
        classes = defaultdict(int)
        superclasses = defaultdict(int)
        raw_labels = defaultdict(int)
        splits = defaultdict(int)
        split_by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        session_coverage: dict[str, dict[str, Any]] = {}
        representative_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        warnings: list[dict[str, Any]] = []
        uncertain_count = 0
        review_required_count = 0
        calibration_reference_count = 0
        empty_scene_count = 0
        processing_complete = 0
        feature_complete = 0
        labeled_count = 0
        validation_states = defaultdict(int)
        split_strategy = str(((ml_set.get("semantics") or {}).get("split_strategy") or "physical_object_id"))
        label_schema_version = "unknown"

        for row in kept_rows:
            take_meta = row.get("take_metadata") if isinstance(row.get("take_metadata"), dict) else {}
            take_summary = row.get("take_summary") if isinstance(row.get("take_summary"), dict) else {}
            normalized = str(row.get("expected_class") or row.get("expected_label") or take_meta.get("expected_class") or "").strip()
            if not normalized:
                normalized = str((take_meta.get("semantic_labels") or [""])[0] or "").strip()
            superclass = str(row.get("expected_subclass") or (take_meta.get("superclass_labels") or [""])[0] or "UNKNOWN").strip() or "UNKNOWN"
            raw_label = str(row.get("raw_label") or row.get("expected_label") or "").strip() or "UNKNOWN"
            split = str(row.get("split") or "unassigned")
            is_trainable = bool(row.get("trainable", row.get("default_trainable", False)))
            session_id = str(take_meta.get("session_id") or take_summary.get("experiment_session_id") or "")
            label_schema_version = str(take_meta.get("normalization_version") or label_schema_version or "unknown")
            validation_state = str(take_meta.get("validation_status") or "unreviewed")
            validation_states[validation_state] += 1

            if normalized:
                labeled_count += 1
                classes[normalized] += 1
                if is_trainable:
                    split_by_class[normalized][split] += 1
            superclasses[superclass] += 1
            raw_labels[raw_label] += 1
            if is_trainable:
                splits[split] += 1
            if normalized == "CALIBRATION_CUBE":
                calibration_reference_count += 1
            if normalized == "EMPTY_SCENE":
                empty_scene_count += 1
            if normalized.endswith("UNCERTAIN") or validation_state == "needs_review":
                uncertain_count += 1
            if bool(row.get("review_required", False)):
                review_required_count += 1
            if bool(take_summary.get("has_done") or take_summary.get("has_ready")):
                processing_complete += 1
            if take_summary.get("processed_class_label") is not None:
                feature_complete += 1

            session_bucket = session_coverage.setdefault(session_id or "unassigned", {
                "session_id": session_id or "unassigned",
                "included_takes": 0,
                "class_composition": defaultdict(int),
                "uncertain_labels": 0,
            })
            session_bucket["included_takes"] += 1
            session_bucket["class_composition"][normalized or "UNLABELED"] += 1
            if normalized.endswith("UNCERTAIN") or validation_state == "needs_review":
                session_bucket["uncertain_labels"] += 1

            if normalized and len(representative_by_class[normalized]) < 4:
                representative_by_class[normalized].append({
                    "take_id": take_summary.get("take_id"),
                    "thumbnail_path": take_summary.get("thumbnail_path"),
                    "validation_status": validation_state,
                    "expected_class": normalized,
                    "session_id": session_id,
                })

        split_integrity = self._build_split_integrity(trainable_rows)
        warnings.extend(split_integrity["warnings"])
        warnings.extend(self._build_balance_warnings(classes, uncertain_count, len(trainable_rows), splits))

        total = len(kept_rows)
        readiness = {
            "labeled_coverage_pct": round((labeled_count / total) * 100, 2) if total else 0.0,
            "split_completeness_pct": round(((total - splits.get("unassigned", 0)) / total) * 100, 2) if total else 0.0,
            "leakage_risk": "INVALID" if split_integrity["leaked_object_ids"] else "NONE",
            "class_imbalance_ratio": self._imbalance_ratio(classes),
            "uncertain_label_count": uncertain_count,
            "take_count": len(kept_rows),
            "physical_object_count": len({str(row.get("physical_object_id") or "") for row in memberships if str(row.get("physical_object_id") or "")}),
            "kept_in_ml_set_count": len(kept_rows),
            "default_trainable_count": len(trainable_rows),
            "trainable_object_count": len(trainable_object_ids),
            "review_required_count": review_required_count,
            "review_required_object_count": len(review_required_object_ids),
            "excluded_count": len(excluded_rows),
            "empty_scene_count": empty_scene_count,
            "calibration_reference_count": calibration_reference_count,
            "processing_completeness": f"{processing_complete}/{total}",
            "feature_coverage": f"{feature_complete}/{total}",
            "split_health": "VALID" if not split_integrity["leaked_object_ids"] else "INVALID",
            "split_status": "assigned" if total and splits.get("unassigned", 0) == 0 else "unassigned",
        }
        return {
            "ml_set": ml_set,
            "dataset": dataset,
            "identity": {
                "take_count": len(kept_rows),
                "physical_object_count": len({str(row.get("physical_object_id") or "") for row in memberships if str(row.get("physical_object_id") or "")}),
                "kept_in_ml_set_count": len(kept_rows),
                "default_trainable_count": len(trainable_rows),
                "trainable_object_count": len(trainable_object_ids),
                "review_required_count": review_required_count,
                "review_required_object_count": len(review_required_object_ids),
                "excluded_count": len(excluded_rows),
                "class_count": len(classes),
                "split_strategy": split_strategy,
                "label_schema_version": label_schema_version,
                "leakage_safe": not split_integrity["leaked_object_ids"],
                "validated_count": validation_states.get("valid", 0) + validation_states.get("validated", 0),
                "unvalidated_count": total - (validation_states.get("valid", 0) + validation_states.get("validated", 0)),
                "ingestion_run": str((ml_set.get("semantics") or {}).get("ingestion_run_id") or ""),
            },
            "readiness": readiness,
            "raw_label_distribution": dict(sorted(raw_labels.items(), key=lambda item: (item[0]))),
            "class_distribution": dict(sorted(classes.items(), key=lambda item: (item[0]))),
            "superclass_distribution": dict(sorted(superclasses.items(), key=lambda item: (item[0]))),
            "split_by_class": {key: dict(value) for key, value in sorted(split_by_class.items(), key=lambda item: item[0])},
            "representative_samples": dict(sorted(representative_by_class.items(), key=lambda item: item[0])),
            "split_summary": split_integrity,
            "source_session_coverage": [
                {
                    **value,
                    "class_composition": dict(sorted(value["class_composition"].items(), key=lambda item: item[0])),
                }
                for key, value in sorted(session_coverage.items(), key=lambda item: item[0])
            ],
            "membership_definition": {
                "membership_mode": str(membership_overview.get("membership_mode") or "ids"),
                "filter_snapshot": (ml_set.get("membership") or {}).get("filter_snapshot") or {},
                "exclude_take_ids": (ml_set.get("membership") or {}).get("exclude_take_ids") or [],
                "semantics": ml_set.get("semantics") or {},
            },
            "warnings": warnings,
            "derived_tasks": self._build_derived_tasks(classes),
            "training_compatibility": {
                "feature_families": {
                    "25D_features": f"{feature_complete}/{total}",
                    "segmentation_masks": "0/0",
                    "geometry_ready": "partial" if processing_complete and processing_complete < total else ("yes" if processing_complete else "no"),
                },
                "classifier_ready": feature_complete == total and total > 0,
                "processing_complete_count": processing_complete,
            },
            "exports": self._export_manifest(resolved_dataset_id, ml_set_id),
        }

    def _export_manifest(self, dataset_id: str, ml_set_id: str) -> dict[str, str]:
        base = self.settings.data_dir / "datasets" / f"dataset_{dataset_id}" / "ml_sets" / f"ml_set_{ml_set_id}"
        materialized = self.settings.data_dir / "ml_sets" / ml_set_id
        return {
            "manifest_csv": str((materialized / "label_manifest.csv") if (materialized / "label_manifest.csv").is_file() else (base / "memberships.json")),
            "label_schema": str((materialized / "label_schema.json") if (materialized / "label_schema.json").is_file() else (base / "ml_set.json")),
        }

    def _build_split_integrity(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        splits = defaultdict(int)
        group_splits: dict[str, set[str]] = defaultdict(set)
        session_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            split = str(row.get("split") or "unassigned")
            obj_id = str(row.get("physical_object_id") or "")
            take_meta = row.get("take_metadata") if isinstance(row.get("take_metadata"), dict) else {}
            session_id = str(take_meta.get("session_id") or "unassigned")
            splits[split] += 1
            session_by_split[split][session_id] += 1
            if obj_id:
                group_splits[obj_id].add(split)
        leaked = sorted([obj_id for obj_id, split_set in group_splits.items() if len(split_set) > 1])
        warnings = []
        if leaked:
            warnings.append({
                "severity": "invalid",
                "code": "object_group_leakage",
                "explanation": "The same physical_object_id appears in multiple splits.",
                "affected_count": len(leaked),
                "affected_ids": leaked[:20],
            })
        return {
            "split_counts": dict(sorted(splits.items(), key=lambda item: item[0])),
            "session_composition_by_split": {key: dict(sorted(value.items(), key=lambda item: item[0])) for key, value in sorted(session_by_split.items(), key=lambda item: item[0])},
            "leaked_object_ids": leaked,
            "warnings": warnings,
        }

    def _build_balance_warnings(self, classes: dict[str, int], uncertain_count: int, total: int, splits: dict[str, int]) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        if classes:
            minority = min(classes.values())
            if minority <= 2:
                warnings.append({
                    "severity": "warning",
                    "code": "tiny_minority_class",
                    "explanation": "One or more classes have very small representation.",
                    "affected_count": minority,
                })
        if uncertain_count:
            warnings.append({
                "severity": "warning" if uncertain_count < max(5, total // 4) else "invalid",
                "code": "uncertain_labels",
                "explanation": "The ML set contains uncertain or review-required labels.",
                "affected_count": uncertain_count,
            })
        if total and splits.get("validation", 0) == 0:
            warnings.append({
                "severity": "warning",
                "code": "missing_validation_split",
                "explanation": "No validation split is currently assigned.",
                "affected_count": 0,
            })
        return warnings

    def _build_derived_tasks(self, classes: dict[str, int]) -> list[dict[str, Any]]:
        return [
            {
                "task_id": "ball_presence_v1",
                "included_classes": sorted([key for key in classes if key.startswith("BALL_") or key == "EMPTY_SCENE"]),
                "excluded_classes": ["REVIEW_REQUIRED"],
                "effective_sample_count": sum(classes.values()),
                "uncertainty_policy": "exclude_review_required",
                "empty_scene_policy": "include_optional",
            },
            {
                "task_id": "ball_vs_non_ball_v1",
                "included_classes": sorted([key for key in classes if key.startswith("BALL_") or key in {"SCRAP_METAL", "EMPTY_SCENE"}]),
                "excluded_classes": ["REVIEW_REQUIRED"],
                "effective_sample_count": sum(classes.values()),
                "uncertainty_policy": "exclude_review_required",
                "empty_scene_policy": "include_optional",
            },
            {
                "task_id": "mining_ball_condition_v1",
                "included_classes": sorted([key for key in classes if key.startswith("BALL_")]),
                "excluded_classes": ["SCRAP_METAL", "CALIBRATION_CUBE", "EMPTY_SCENE", "REVIEW_REQUIRED"],
                "effective_sample_count": sum(value for key, value in classes.items() if key.startswith("BALL_")),
                "uncertainty_policy": "exclude_uncertain_recommended",
                "empty_scene_policy": "exclude",
            },
        ]

    @staticmethod
    def _imbalance_ratio(classes: dict[str, int]) -> float:
        values = [value for value in classes.values() if value > 0]
        if not values:
            return 0.0
        return round(max(values) / max(1, min(values)), 2)

    def export_file(self, ml_set_id: str, kind: str, dataset_id: str | None = None) -> Path:
        ml_set, resolved_dataset_id = self._resolve_ml_set(ml_set_id, dataset_id)
        base = self.settings.data_dir / "datasets" / f"dataset_{resolved_dataset_id}" / "ml_sets" / f"ml_set_{ml_set_id}"
        materialized = self.settings.data_dir / "ml_sets" / ml_set_id
        if kind == "manifest":
            candidate = materialized / "label_manifest.csv"
            return candidate if candidate.is_file() else base / "memberships.json"
        if kind == "splits":
            candidate = materialized / "split_manifest.csv"
            return candidate if candidate.is_file() else base / "memberships.json"
        if kind == "label_schema":
            candidate = materialized / "label_schema.json"
            return candidate if candidate.is_file() else base / "ml_set.json"
        if kind == "tasks":
            candidate = materialized / "tasks.json"
            return candidate if candidate.is_file() else base / "ml_set.json"
        if kind == "snapshot":
            candidate = materialized / "ml_set.json"
            return candidate if candidate.is_file() else base / "ml_set.json"
        raise ValueError(f"Unknown export kind: {kind}")
