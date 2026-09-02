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


def _row_is_trainable(row: dict[str, Any], *, default: bool = False) -> bool:
    for key in ("trainable", "default_trainable"):
        value = row.get(key)
        if value is not None:
            return bool(value)
    return default


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
        trainable_rows = [row for row in kept_rows if _row_is_trainable(row)]
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
            is_trainable = _row_is_trainable(row)
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

        split_integrity = self._build_split_integrity(kept_rows)
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

    def list_members(
        self,
        ml_set_id: str,
        dataset_id: str | None = None,
        *,
        physical_object_id: str | None = None,
        raw_label: str | None = None,
        normalized_class: str | None = None,
        superclass: str | None = None,
        membership_status: str | None = None,
        split: str | None = None,
        session_id: str | None = None,
        search: str | None = None,
        validation_status: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        ml_set, resolved_dataset_id, memberships = self._memberships_with_metadata(ml_set_id, dataset_id)
        dataset = self.dataset_service.get_dataset(resolved_dataset_id) or {"id": resolved_dataset_id, "name": resolved_dataset_id}
        items = [self._build_member_row(resolved_dataset_id, row) for row in memberships]
        filtered = self._filter_member_items(
            items,
            physical_object_id=physical_object_id,
            raw_label=raw_label,
            normalized_class=normalized_class,
            superclass=superclass,
            membership_status=membership_status,
            split=split,
            session_id=session_id,
            search=search,
            validation_status=validation_status,
        )
        filtered.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("take_id") or "")), reverse=True)
        resolved_limit = max(1, min(int(limit), 500))
        resolved_offset = max(0, int(offset))
        page_items = filtered[resolved_offset : resolved_offset + resolved_limit]
        next_offset = resolved_offset + len(page_items)
        object_ids = {
            str(item.get("physical_object_id") or "").strip()
            for item in filtered
            if str(item.get("physical_object_id") or "").strip()
        }
        review_required_count = sum(1 for item in filtered if bool(item.get("review_required")))
        return {
            "ml_set": ml_set,
            "dataset": dataset,
            "items": page_items,
            "filtered_count": len(filtered),
            "total_count": len(items),
            "limit": resolved_limit,
            "offset": resolved_offset,
            "has_more": next_offset < len(filtered),
            "next_offset": next_offset if next_offset < len(filtered) else None,
            "stats": {
                "member_count": len(filtered),
                "object_count": len(object_ids),
                "review_required_count": review_required_count,
            },
        }

    def build_member_facets(
        self,
        ml_set_id: str,
        dataset_id: str | None = None,
        *,
        physical_object_id: str | None = None,
        raw_label: str | None = None,
        normalized_class: str | None = None,
        superclass: str | None = None,
        membership_status: str | None = None,
        split: str | None = None,
        session_id: str | None = None,
        search: str | None = None,
        validation_status: str | None = None,
    ) -> dict[str, Any]:
        ml_set, resolved_dataset_id, memberships = self._memberships_with_metadata(ml_set_id, dataset_id)
        dataset = self.dataset_service.get_dataset(resolved_dataset_id) or {"id": resolved_dataset_id, "name": resolved_dataset_id}
        items = [self._build_member_row(resolved_dataset_id, row) for row in memberships]
        fully_filtered = self._filter_member_items(
            items,
            physical_object_id=physical_object_id,
            raw_label=raw_label,
            normalized_class=normalized_class,
            superclass=superclass,
            membership_status=membership_status,
            split=split,
            session_id=session_id,
            search=search,
            validation_status=validation_status,
        )

        # Picker facets (object/split/membership-status/session) always cover
        # the full, unfiltered member list so the UI can navigate to any
        # value regardless of the currently active filters. Descriptive
        # facets (raw label, normalized class, superclass) summarize the
        # currently filtered result set instead.
        object_scope = items
        raw_label_scope = fully_filtered
        normalized_scope = fully_filtered
        superclass_scope = fully_filtered
        split_scope = items
        membership_scope = items
        session_scope = items

        object_rows: dict[str, dict[str, Any]] = {}
        for item in object_scope:
            value = str(item.get("physical_object_id") or "").strip()
            if not value:
                continue
            bucket = object_rows.setdefault(value, {
                "value": value,
                "take_count": 0,
                "normalized_class": str(item.get("normalized_class") or ""),
                "superclass": str(item.get("superclass") or ""),
            })
            bucket["take_count"] = int(bucket["take_count"]) + 1
            if not bucket["normalized_class"] and item.get("normalized_class"):
                bucket["normalized_class"] = str(item.get("normalized_class") or "")
            if not bucket["superclass"] and item.get("superclass"):
                bucket["superclass"] = str(item.get("superclass") or "")

        def count_values(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
            counts: dict[str, int] = defaultdict(int)
            for item in rows:
                value = str(item.get(field) or "").strip()
                if not value:
                    continue
                counts[value] += 1
            return [{"value": value, "count": count} for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]

        session_rows: dict[str, dict[str, Any]] = {}
        for item in session_scope:
            value = str(item.get("session_id") or "").strip()
            if not value:
                continue
            bucket = session_rows.setdefault(value, {"value": value, "name": str(item.get("session_name") or value), "count": 0})
            bucket["count"] = int(bucket["count"]) + 1

        membership_counts: dict[str, int] = {
            "all": len(membership_scope),
            "default_trainable": 0,
            "review_required": 0,
            "excluded": 0,
            "uncertain": 0,
        }
        for item in membership_scope:
            if bool(item.get("default_trainable")):
                membership_counts["default_trainable"] += 1
            if bool(item.get("review_required")):
                membership_counts["review_required"] += 1
            if not bool(item.get("include", True)):
                membership_counts["excluded"] += 1
            if bool(item.get("is_uncertain")):
                membership_counts["uncertain"] += 1

        object_id_options = sorted(object_rows.values(), key=lambda item: _natural_sort_key(str(item["value"])))
        return {
            "ml_set": ml_set,
            "dataset": dataset,
            "stats": {
                "member_count": len(fully_filtered),
                "object_count": len({str(item.get("physical_object_id") or "").strip() for item in fully_filtered if str(item.get("physical_object_id") or "").strip()}),
            },
            "object_ids": object_id_options,
            "raw_labels": count_values(raw_label_scope, "raw_label"),
            "normalized_classes": count_values(normalized_scope, "normalized_class"),
            "superclasses": count_values(superclass_scope, "superclass"),
            "splits": count_values(split_scope, "split"),
            "membership_statuses": [{"value": value, "count": count} for value, count in membership_counts.items() if count > 0 or value == "all"],
            "sessions": sorted(session_rows.values(), key=lambda item: (-int(item["count"]), str(item["name"]))),
        }

    def _build_member_row(self, dataset_id: str, row: dict[str, Any]) -> dict[str, Any]:
        take_meta = row.get("take_metadata") if isinstance(row.get("take_metadata"), dict) else {}
        take_summary = row.get("take_summary") if isinstance(row.get("take_summary"), dict) else {}
        physical_object_id = str(row.get("physical_object_id") or take_meta.get("physical_object_id") or "").strip()
        normalized = str(
            row.get("expected_class")
            or take_meta.get("expected_class")
            or take_meta.get("normalized_class")
            or ((take_meta.get("semantic_labels") or [""])[0] if isinstance(take_meta.get("semantic_labels"), list) else "")
            or ""
        ).strip()
        resolved_superclass = str(
            row.get("expected_subclass")
            or ((take_meta.get("superclass_labels") or [""])[0] if isinstance(take_meta.get("superclass_labels"), list) else "")
            or "UNKNOWN"
        ).strip() or "UNKNOWN"
        raw_label = str(
            row.get("raw_label")
            or row.get("expected_label")
            or take_meta.get("raw_operator_label")
            or ""
        ).strip()
        include = bool(row.get("include", True))
        default_trainable = bool(row.get("default_trainable", False))
        trainable = _row_is_trainable(row, default=include)
        review_required = bool(row.get("review_required", False))
        split = str(row.get("split") or "unassigned").strip() or "unassigned"
        validation_state = str(take_meta.get("validation_status") or take_summary.get("validation_status") or "unreviewed").strip() or "unreviewed"
        notes = str(row.get("notes") or "").strip()
        source_row = str(row.get("source_row") or "").strip()
        is_uncertain = review_required or validation_state == "needs_review" or normalized.upper().endswith("UNCERTAIN")
        warnings: list[str] = []
        if review_required:
            warnings.append("review_required")
        if not physical_object_id:
            warnings.append("missing_physical_object_id")
        if split == "unassigned":
            warnings.append("missing_split")
        if not include:
            warnings.append("excluded_from_training")
        if validation_state not in {"valid", "validated"}:
            warnings.append("not_validated")
        processing_by_family = take_summary.get("processing_by_family") if isinstance(take_summary.get("processing_by_family"), list) else []
        feature_ready = bool(take_summary.get("processed_class_label")) or any(
            isinstance(item, dict) and bool(item.get("hasCompletedOutput")) for item in processing_by_family
        )
        return {
            "dataset_id": dataset_id,
            "ml_set_id": str(row.get("ml_set_id") or ""),
            "take_id": str(row.get("take_id") or ""),
            "take_name": str(take_summary.get("friendly_name") or row.get("take_id") or ""),
            "created_at": take_summary.get("created_at"),
            "thumbnail_path": take_summary.get("thumbnail_path"),
            "physical_object_id": physical_object_id or None,
            "raw_label": raw_label or None,
            "normalized_class": normalized or None,
            "superclass": resolved_superclass,
            "review_required": review_required,
            "default_trainable": default_trainable,
            "trainable": trainable,
            "include": include,
            "split": split,
            "validation_status": validation_state,
            "session_id": str(take_summary.get("experiment_session_id") or take_meta.get("session_id") or "") or None,
            "session_name": str(take_summary.get("experiment_session_name") or "") or None,
            "notes": notes or None,
            "source_row": source_row or None,
            "is_uncertain": is_uncertain,
            "processing_ready": bool(take_summary.get("has_done") or take_summary.get("has_ready")),
            "feature_ready": feature_ready,
            "processed_class_label": str(take_summary.get("processed_class_label") or "") or None,
            "processed_superclass": str(take_summary.get("processed_superclass") or "") or None,
            "warnings": warnings,
            "warning_count": len(warnings),
            "label_provenance": {
                "raw_label": raw_label or None,
                "expected_label": row.get("expected_label"),
                "normalized_class": normalized or None,
                "superclass": resolved_superclass,
                "normalization_version": row.get("normalization_version") or take_meta.get("normalization_version"),
                "source_row": row.get("source_row"),
                "label_policy": row.get("label_policy"),
            },
            "feature_summary": {
                "processed_class_label": str(take_summary.get("processed_class_label") or "") or None,
                "processed_superclass": str(take_summary.get("processed_superclass") or "") or None,
                "processing_by_family": processing_by_family,
            },
        }

    @staticmethod
    def _matches_member_filters(
        item: dict[str, Any],
        *,
        physical_object_id: str | None = None,
        raw_label: str | None = None,
        normalized_class: str | None = None,
        superclass: str | None = None,
        membership_status: str | None = None,
        split: str | None = None,
        session_id: str | None = None,
        search: str | None = None,
        validation_status: str | None = None,
    ) -> bool:
        if physical_object_id and str(item.get("physical_object_id") or "").strip().lower() != physical_object_id.strip().lower():
            return False
        if raw_label and raw_label.strip().lower() != str(item.get("raw_label") or "").strip().lower():
            return False
        if normalized_class and normalized_class.strip().lower() != str(item.get("normalized_class") or "").strip().lower():
            return False
        if superclass and superclass.strip().lower() != str(item.get("superclass") or "").strip().lower():
            return False
        if split and split.strip().lower() != str(item.get("split") or "").strip().lower():
            return False
        if session_id and session_id.strip().lower() != str(item.get("session_id") or "").strip().lower():
            return False
        if validation_status and validation_status.strip().lower() != str(item.get("validation_status") or "").strip().lower():
            return False
        if search:
            query = search.strip().lower()
            haystack = " ".join(
                [
                    str(item.get("take_id") or ""),
                    str(item.get("take_name") or ""),
                    str(item.get("physical_object_id") or ""),
                    str(item.get("raw_label") or ""),
                    str(item.get("normalized_class") or ""),
                    str(item.get("superclass") or ""),
                    str(item.get("session_name") or ""),
                    str(item.get("notes") or ""),
                    str(item.get("source_row") or ""),
                ]
            ).lower()
            if query not in haystack:
                return False
        if membership_status:
            value = membership_status.strip().lower()
            if value == "review_required" and not bool(item.get("review_required")):
                return False
            if value == "default_trainable" and not bool(item.get("default_trainable")):
                return False
            if value == "trainable" and not bool(item.get("trainable")):
                return False
            if value == "excluded" and bool(item.get("include", True)):
                return False
            if value == "non_trainable" and bool(item.get("trainable")):
                return False
            if value == "uncertain" and not bool(item.get("is_uncertain")):
                return False
        return True

    def _filter_member_items(
        self,
        items: list[dict[str, Any]],
        *,
        physical_object_id: str | None = None,
        raw_label: str | None = None,
        normalized_class: str | None = None,
        superclass: str | None = None,
        membership_status: str | None = None,
        split: str | None = None,
        session_id: str | None = None,
        search: str | None = None,
        validation_status: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in items
            if self._matches_member_filters(
                item,
                physical_object_id=physical_object_id,
                raw_label=raw_label,
                normalized_class=normalized_class,
                superclass=superclass,
                membership_status=membership_status,
                split=split,
                session_id=session_id,
                search=search,
                validation_status=validation_status,
            )
        ]

    def _export_manifest(self, dataset_id: str, ml_set_id: str) -> dict[str, str]:
        materialized = self.settings.data_dir / "ml_sets" / ml_set_id
        manifest = materialized / "label_manifest.csv"
        schema = materialized / "label_schema.json"
        return {
            "manifest_csv": str(manifest) if manifest.is_file() else f"sqlite:ml_set_membership/{dataset_id}/{ml_set_id}",
            "label_schema": str(schema) if schema.is_file() else f"sqlite:ml_set/{dataset_id}/{ml_set_id}",
        }

    def _memberships_fallback(self, dataset_id: str, ml_set_id: str) -> Path:
        """Write the stored memberships out so a download has a file to serve.

        The fallback used to point at data/datasets/.../memberships.json. That
        document is a table now, so the path would dangle; this regenerates it
        from the catalog next to the materialized artifacts instead.
        """
        from vision_3d_acquisition.datasets import DatasetService

        target = self.settings.data_dir / "ml_sets" / ml_set_id / "memberships.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = DatasetService(self.settings.data_dir).docs.read_memberships(dataset_id, ml_set_id)
        target.write_text(json.dumps({"memberships": rows}, indent=2), encoding="utf-8")
        return target

    def _ml_set_fallback(self, dataset_id: str, ml_set_id: str) -> Path:
        """Write the stored ml_set document out so a download has a file to serve.

        label_schema/tasks/snapshot used to fall back to a `base` variable that
        pointed at data/datasets/.../ml_set.json — a leftover from before that
        document became a row, and no longer assigned anywhere in this method,
        so hitting this fallback raised NameError instead of a stale 404. This
        regenerates the file from the catalog instead.
        """
        from vision_3d_acquisition.datasets import DatasetService

        target = self.settings.data_dir / "ml_sets" / ml_set_id / "ml_set.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = DatasetService(self.settings.data_dir).get_ml_set(dataset_id, ml_set_id) or {}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

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
        materialized = self.settings.data_dir / "ml_sets" / ml_set_id
        if kind == "manifest":
            candidate = materialized / "label_manifest.csv"
            return candidate if candidate.is_file() else self._memberships_fallback(resolved_dataset_id, ml_set_id)
        if kind == "splits":
            candidate = materialized / "split_manifest.csv"
            return candidate if candidate.is_file() else self._memberships_fallback(resolved_dataset_id, ml_set_id)
        if kind == "label_schema":
            candidate = materialized / "label_schema.json"
            return candidate if candidate.is_file() else self._ml_set_fallback(resolved_dataset_id, ml_set_id)
        if kind == "tasks":
            candidate = materialized / "tasks.json"
            return candidate if candidate.is_file() else self._ml_set_fallback(resolved_dataset_id, ml_set_id)
        if kind == "snapshot":
            candidate = materialized / "ml_set.json"
            return candidate if candidate.is_file() else self._ml_set_fallback(resolved_dataset_id, ml_set_id)
        raise ValueError(f"Unknown export kind: {kind}")


def _natural_sort_key(value: str) -> tuple[Any, ...]:
    output: list[Any] = []
    token = ""
    is_digit = False
    for char in value:
        if char.isdigit():
            if token and not is_digit:
                output.append(token.lower())
                token = ""
            token += char
            is_digit = True
            continue
        if token and is_digit:
            output.append(int(token))
            token = ""
        token += char
        is_digit = False
    if token:
        output.append(int(token) if is_digit else token.lower())
    return tuple(output)
