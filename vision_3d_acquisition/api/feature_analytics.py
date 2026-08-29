from __future__ import annotations

from datetime import datetime
from statistics import mean, pstdev
import time
from typing import Any

from vision_3d_acquisition.api.filesystem import list_take_ids, read_json
from vision_3d_acquisition.api.feature_catalog import FEATURE_REGISTRY, feature_definition_for_key
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.label_normalization import LabelNormalizationService
from vision_3d_acquisition.ml.features.surface_sphere_fit_workflow import load_feature_backfill_index
from vision_3d_acquisition.vision_core.geometry.sphere_consistency_features import derive_sphere_consistency_from_object
from vision_3d_acquisition.ml.label_normalization import resolve_audit_superclass
from vision_3d_acquisition.ml.ml_set_summary import MLSetSummaryService


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:
        return None
    return num


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = str(value).strip()
    if not parsed:
        return None
    if parsed.endswith("Z"):
        parsed = parsed[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(parsed)
    except ValueError:
        return None


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _split_csv(value)
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                items.extend(_split_csv(entry))
            elif entry is not None:
                text = str(entry).strip()
                if text:
                    items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _first_string(*values: Any) -> str | None:
    for value in values:
        items = _string_list(value)
        if items:
            return items[0]
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _classify_superclass(raw_object: dict[str, Any], fallback: str | None = None) -> str:
    superclass = str(raw_object.get("superclass") or "").strip().upper()
    if superclass:
        return superclass
    if fallback:
        return str(fallback).strip().upper()
    label = str(raw_object.get("class_name") or raw_object.get("label") or "").strip().upper()
    if label.startswith("BALL_GOOD"):
        return "BALL_GOOD"
    if label.startswith("BALL_SCRAP"):
        return "BALL_SCRAP"
    if label.startswith("SCRAP"):
        return "SCRAP"
    return "UNKNOWN"


def _extract_feature_values(raw_object: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    out: dict[str, float] = {}
    sources: dict[str, str] = {}
    for key, value in raw_object.items():
        if key.startswith("feature_"):
            parsed = _to_float(value)
            if parsed is not None:
                out[key] = parsed
                sources[key] = "pipeline_run"
    for feature_key, definition in FEATURE_REGISTRY.items():
        if feature_key in out:
            continue
        for path in definition.extraction_paths:
            parsed = _extract_numeric_path(raw_object, path)
            if parsed is not None:
                out[feature_key] = parsed
                sources[feature_key] = "pipeline_run"
                break
    roundness = _to_float(raw_object.get("feature_footprint_roundness") or raw_object.get("footprint_roundness") or raw_object.get("sphericity_score"))
    if roundness is not None and "footprint_roundness" not in out:
        out["footprint_roundness"] = roundness
        sources["footprint_roundness"] = "pipeline_run"
    derived = derive_sphere_consistency_from_object(raw_object)
    for key, value in derived.items():
        if key not in out and value is not None:
            out[key] = value
            sources[key] = "derived"
    return out, sources


def _merge_backfill_features(
    features: dict[str, float],
    feature_sources: dict[str, str],
    *,
    backfill_index: dict[str, dict[str, Any]],
    object_id: str,
) -> tuple[dict[str, float], dict[str, str]]:
    if not backfill_index:
        return features, feature_sources
    entry = backfill_index.get(str(object_id))
    if not entry:
        return features, feature_sources
    feature_key = str(entry.get("feature_key") or "")
    value = _to_float(entry.get("value"))
    if not feature_key or value is None or feature_key in features:
        return features, feature_sources
    merged = dict(features)
    merged[feature_key] = value
    sources = dict(feature_sources)
    sources[feature_key] = str(entry.get("source") or "feature_backfill")
    return merged, sources


def _extract_numeric_path(raw_object: dict[str, Any], path: str) -> float | None:
    if not path:
        return None
    if "." not in path:
        return _to_float(raw_object.get(path))
    current: Any = raw_object
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _to_float(current)


def _resolve_labeled_superclass(
    *,
    membership_meta: dict[str, Any] | None,
    take_management: dict[str, Any],
    raw_labels: list[str],
    normalized_class_candidates: list[str],
    label_normalizer: LabelNormalizationService | None,
) -> tuple[str, str | None]:
    membership_explicit = _first_string(
        (membership_meta or {}).get("expected_subclass"),
        (membership_meta or {}).get("superclass"),
        (membership_meta or {}).get("expected_superclass"),
    )
    resolved_membership = resolve_audit_superclass(membership_explicit)
    if resolved_membership != "UNKNOWN":
        return resolved_membership, membership_explicit

    take_superclass = _first_string(take_management.get("superclass_labels"))
    resolved_take_superclass = resolve_audit_superclass(take_superclass)
    if resolved_take_superclass != "UNKNOWN":
        return resolved_take_superclass, take_superclass

    for normalized_class in normalized_class_candidates:
        resolved_normalized = resolve_audit_superclass(normalized_class)
        if resolved_normalized != "UNKNOWN":
            return resolved_normalized, normalized_class

    if label_normalizer is not None and raw_labels:
        normalized = label_normalizer.normalize_tags(raw_labels)
        normalized_candidates = _unique_preserve_order(
            [str(normalized.normalized_class or "").strip()] + [str(item).strip() for item in (normalized.semantic_labels or []) if str(item).strip()]
        )
        for candidate in normalized_candidates:
            resolved_taxonomy = resolve_audit_superclass(candidate)
            if resolved_taxonomy != "UNKNOWN":
                return resolved_taxonomy, candidate
        for candidate in normalized.superclass_labels:
            resolved_taxonomy_superclass = resolve_audit_superclass(candidate)
            if resolved_taxonomy_superclass != "UNKNOWN":
                return resolved_taxonomy_superclass, candidate

    raw_label_fallback = _first_string(
        raw_labels,
        (membership_meta or {}).get("raw_label"),
        (membership_meta or {}).get("expected_label"),
        take_management.get("raw_operator_label"),
    )
    resolved_raw_label = resolve_audit_superclass(raw_label_fallback)
    if resolved_raw_label != "UNKNOWN":
        return resolved_raw_label, raw_label_fallback

    return "UNKNOWN", None


def _feature_definitions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: set[str] = set()
    for record in records:
        for key in (record.get("features") or {}).keys():
            found.add(str(key))
    keys = sorted(set(FEATURE_REGISTRY) | found)
    return [feature_definition_for_key(key).payload() for key in keys]


def build_feature_records(
    settings: ApiSettings,
    *,
    query: dict[str, Any] | None = None,
    max_takes: int = 600,
    time_budget_ms: int = 2500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    filters = query or {}
    dataset_ids = set(_string_list(filters.get("dataset_ids") or filters.get("datasets") or filters.get("dataset_id")))
    session_ids = set(_string_list(filters.get("session_ids") or filters.get("sessions") or filters.get("session_id")))
    raw_labels_filter = {item.lower() for item in _string_list(filters.get("raw_labels") or filters.get("labels"))}
    normalized_classes_filter = {item.lower() for item in _string_list(filters.get("normalized_classes") or filters.get("normalized_class"))}
    labeled_superclass_filter = {item.upper() for item in _string_list(filters.get("labeled_superclasses"))}
    processed_superclass_filter = {
        item.upper()
        for item in _string_list(filters.get("processed_superclasses") or filters.get("superclasses") or filters.get("superclass"))
    }
    processed_class_filter = {item.lower() for item in _string_list(filters.get("processed_classes"))}
    physical_object_filter = {item.strip() for item in _string_list(filters.get("physical_object_ids") or filters.get("physical_object_id")) if item.strip()}
    take_id_filter = {item.strip() for item in _string_list(filters.get("take_ids") or filters.get("take_id")) if item.strip()}
    validation_filter = {item.lower() for item in _string_list(filters.get("validation_status"))}
    split_filter = {item.lower() for item in _string_list(filters.get("split"))}
    pipeline_filter = set(_string_list(filters.get("pipeline_ids") or filters.get("pipeline_id") or filters.get("pipeline")))
    calibration_filter = set(_string_list(filters.get("calibration_ids") or filters.get("calibration_id") or filters.get("calibration")))
    selected_features = set(_string_list(filters.get("feature_selection")))
    from_date = _parse_date(filters.get("date_from"))
    to_date = _parse_date(filters.get("date_to"))
    ml_set_id = str(filters.get("ml_set_id") or "").strip()

    dataset_service = DatasetService(settings.data_dir)
    ml_set_take_ids: set[str] | None = None
    ml_set_membership_by_take: dict[str, dict[str, Any]] = {}
    if ml_set_id:
        resolved_dataset_id = next(iter(dataset_ids), None) if len(dataset_ids) == 1 else _first_string(filters.get("dataset_id"))
        ml_set_service = MLSetSummaryService(settings)
        try:
            membership_payload = ml_set_service.list_members(ml_set_id, resolved_dataset_id, limit=500, offset=0)
            ml_set_take_ids = set()
            for item in membership_payload.get("items") or []:
                take_id = str(item.get("take_id") or "")
                if not take_id:
                    continue
                ml_set_take_ids.add(take_id)
                ml_set_membership_by_take[take_id] = item
            while membership_payload.get("has_more"):
                membership_payload = ml_set_service.list_members(
                    ml_set_id,
                    resolved_dataset_id,
                    limit=500,
                    offset=int(membership_payload.get("next_offset") or 0),
                )
                for item in membership_payload.get("items") or []:
                    take_id = str(item.get("take_id") or "")
                    if not take_id:
                        continue
                    ml_set_take_ids.add(take_id)
                    ml_set_membership_by_take[take_id] = item
            if not dataset_ids:
                resolved_ml_set = dataset_service.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=resolved_dataset_id)
                resolved_dataset = str(resolved_ml_set.get("dataset_id") or "").strip()
                if resolved_dataset:
                    dataset_ids.add(resolved_dataset)
        except ValueError:
            ml_set_take_ids = set()

    preflight_duration_ms = round((time.monotonic() - started) * 1000.0, 2)
    if ml_set_take_ids is not None:
        take_ids = sorted(ml_set_take_ids, reverse=True)
    elif dataset_ids:
        take_ids = sorted(
            {
                take_id
                for dataset_id in dataset_ids
                for _session_id, take_id, _payload in dataset_service.iter_dataset_takes(dataset_id)
            },
            reverse=True,
        )
    else:
        take_ids = sorted(list_take_ids(settings), reverse=True)

    scan_started = time.monotonic()
    records: list[dict[str, Any]] = []
    try:
        label_normalizer = LabelNormalizationService()
    except Exception:
        label_normalizer = None
    scan_meta = {
        "take_candidates": len(take_ids),
        "takes_scanned": 0,
        "takes_with_result": 0,
        "objects_scanned": 0,
        "records_emitted": 0,
        "stopped_by": None,
        "max_takes": max(1, int(max_takes)),
        "time_budget_ms": max(50, int(time_budget_ms)),
        "preflight_duration_ms": preflight_duration_ms,
    }

    for take_id in take_ids:
        elapsed_ms = (time.monotonic() - scan_started) * 1000.0
        if elapsed_ms >= scan_meta["time_budget_ms"]:
            scan_meta["stopped_by"] = "time_budget"
            break
        if scan_meta["takes_scanned"] >= scan_meta["max_takes"]:
            scan_meta["stopped_by"] = "take_limit"
            break

        scan_meta["takes_scanned"] += 1

        incoming_meta = read_json(settings.incoming_dir / take_id / "metadata.json") or {}
        take_management = dataset_service.load_take_metadata(take_id=take_id, source_metadata=incoming_meta)
        result = read_json(settings.processed_dir / take_id / "result.json") or {}
        if not result:
            continue
        scan_meta["takes_with_result"] += 1

        dataset_id = str(take_management.get("dataset_id") or "") or None
        session_id = str(take_management.get("session_id") or incoming_meta.get("session_id") or "") or None
        validation_status = str(take_management.get("validation_status") or "unreviewed")
        split = str(take_management.get("split") or "")
        created_at = str(incoming_meta.get("created_at") or result.get("processed_at") or "")
        created_date = _parse_date(created_at)

        if dataset_ids and (dataset_id or "") not in dataset_ids:
            continue
        if session_ids and (session_id or "") not in session_ids:
            continue
        if validation_filter and validation_status.lower() not in validation_filter:
            continue
        if ml_set_take_ids is not None and take_id not in ml_set_take_ids:
            continue
        if take_id_filter and take_id not in take_id_filter:
            continue
        if from_date and (created_date is None or created_date < from_date):
            continue
        if to_date and (created_date is None or created_date > to_date):
            continue
        membership_meta = ml_set_membership_by_take.get(take_id) if ml_set_membership_by_take else None
        effective_split = str((membership_meta or {}).get("split") or split or "")
        if split_filter and effective_split.lower() not in split_filter:
            continue
        effective_physical_object_id = str((membership_meta or {}).get("physical_object_id") or take_management.get("physical_object_id") or "").strip()
        if physical_object_filter and effective_physical_object_id not in physical_object_filter:
            continue

        processing_pipeline = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
        pipeline_id = str(processing_pipeline.get("id") or "")
        run_id = str(result.get("run_id") or "")
        calibration_id = str(result.get("calibration_id") or "")

        if pipeline_filter and pipeline_id not in pipeline_filter:
            continue
        if calibration_filter and calibration_id not in calibration_filter:
            continue

        backfill_index = load_feature_backfill_index(settings.processed_dir, take_id, "surface_sphere_fit_rmse_mm")

        objects = [
            item
            for item in ((result.get("objects") or []) + (result.get("rejected_objects") or []))
            if isinstance(item, dict)
        ]

        labeled_superclasses = [resolve_audit_superclass(item) for item in (take_management.get("superclass_labels") or []) if str(item)]
        labeled_superclasses = [item for item in labeled_superclasses if item != "UNKNOWN"]
        fallback_superclass = labeled_superclasses[0] if labeled_superclasses else str(result.get("processed_superclass") or "UNKNOWN")
        raw_labels = _unique_preserve_order([str(item).strip() for item in (take_management.get("labels") or []) if str(item).strip()])
        normalized_class_candidates = _unique_preserve_order(
            [str(take_management.get("normalized_class") or "").strip()]
            + [str(item).strip() for item in (take_management.get("semantic_labels") or []) if str(item).strip()]
            + [str((membership_meta or {}).get("expected_class") or "").strip()]
            + [str((membership_meta or {}).get("normalized_class") or "").strip()]
        )
        normalized_class = normalized_class_candidates[0] if normalized_class_candidates else None
        labeled_superclass, labeled_superclass_source = _resolve_labeled_superclass(
            membership_meta=membership_meta,
            take_management=take_management,
            raw_labels=raw_labels,
            normalized_class_candidates=normalized_class_candidates,
            label_normalizer=label_normalizer,
        )
        labeled_superclass_unresolved = labeled_superclass == "UNKNOWN" and normalized_class is not None

        for raw in objects:
            scan_meta["objects_scanned"] += 1
            features, feature_sources = _extract_feature_values(raw)
            features, feature_sources = _merge_backfill_features(
                features,
                feature_sources,
                backfill_index=backfill_index,
                object_id=str(raw.get("object_id") or ""),
            )
            if selected_features:
                features = {key: value for key, value in features.items() if key in selected_features}
            if not features:
                continue

            canonical_labels = _unique_preserve_order([str(raw.get("class_name") or "").strip()] + normalized_class_candidates)
            if raw_labels_filter and not any(item.lower() in raw_labels_filter for item in raw_labels):
                continue
            if normalized_classes_filter and not any(item.lower() in normalized_classes_filter for item in canonical_labels):
                continue

            processed_superclass = _classify_superclass(raw, fallback_superclass)
            processed_class = str(raw.get("class_name") or raw.get("label") or "").strip() or None
            if labeled_superclass_filter and str(labeled_superclass or "UNKNOWN").upper() not in labeled_superclass_filter:
                continue
            if processed_superclass_filter and processed_superclass not in processed_superclass_filter:
                continue
            if processed_class_filter and str(processed_class or "").lower() not in processed_class_filter:
                continue

            records.append(
                {
                    "object_id": str(raw.get("object_id") or ""),
                    "take_id": take_id,
                    "dataset_id": dataset_id,
                    "session_id": session_id,
                    "physical_object_id": effective_physical_object_id or None,
                    "pipeline_id": pipeline_id,
                    "run_id": run_id,
                    "stage_id": "measurement",
                    "labels": canonical_labels,
                    "raw_labels": raw_labels,
                    "normalized_class": normalized_class,
                    "labeled_superclass": labeled_superclass,
                    "labeled_superclass_source": labeled_superclass_source,
                    "labeled_superclass_unresolved": labeled_superclass_unresolved,
                    "processed_superclass": processed_superclass,
                    "processed_class": processed_class,
                    "processed_label": str(raw.get("label") or "").strip() or None,
                    "confidence": _to_float(raw.get("confidence")),
                    "superclass": processed_superclass,
                    "features": features,
                    "feature_sources": feature_sources,
                    "acquisition_metadata": {
                        "source": incoming_meta.get("source"),
                        "modalities": result.get("input_modalities") or incoming_meta.get("modalities") or [],
                        "session_id": incoming_meta.get("session_id"),
                    },
                    "calibration_metadata": {
                        "calibration_id": calibration_id or None,
                        "calibration_file": result.get("calibration_file"),
                    },
                    "timestamp": created_at or None,
                    "validation_status": validation_status,
                    "split": effective_split or None,
                    "annotation": None,
                }
            )

    scan_meta["records_emitted"] = len(records)
    scan_meta["scan_duration_ms"] = round((time.monotonic() - scan_started) * 1000.0, 2)
    scan_meta["duration_ms"] = round((time.monotonic() - started) * 1000.0, 2)
    return records, scan_meta


def summarize_feature_values(records: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    values = [_to_float((record.get("features") or {}).get(feature_key)) for record in records]
    clean = [value for value in values if value is not None]
    total = len(values)
    missing = total - len(clean)
    if not clean:
        return {"count": 0, "missing": missing, "missing_pct": 100.0 if total else 0.0, "min": None, "max": None, "mean": None, "std": None}
    mu = mean(clean)
    return {
        "count": len(clean),
        "missing": missing,
        "missing_pct": (missing / total) * 100.0 if total else 0.0,
        "min": min(clean),
        "max": max(clean),
        "mean": mu,
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
    }


def build_distributions(
    records: list[dict[str, Any]],
    *,
    feature_key: str,
    group_by: str = "superclass",
    bins: int = 24,
    mode: str = "count",
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        value = _to_float((record.get("features") or {}).get(feature_key))
        if value is None:
            continue
        if group_by == "raw_label":
            key = str(((record.get("raw_labels") or ["UNKNOWN"])[0]))
        elif group_by == "processed_superclass":
            key = str(record.get("processed_superclass") or "UNKNOWN")
        elif group_by == "labeled_superclass":
            key = str(record.get("labeled_superclass") or "UNKNOWN")
        elif group_by == "processed_class":
            key = str(record.get("processed_class") or "UNKNOWN")
        elif group_by == "normalized_class":
            key = str(record.get("normalized_class") or ((record.get("labels") or ["UNKNOWN"])[0]))
        elif group_by == "label":
            key = str(((record.get("labels") or ["UNKNOWN"])[0]))
        elif group_by == "physical_object_id":
            key = str(record.get("physical_object_id") or "UNASSIGNED_OBJECT")
        else:
            key = str(record.get("superclass") or "UNKNOWN")
        grouped.setdefault(key, []).append(value)

    all_values = [value for values in grouped.values() for value in values]
    if not all_values:
        return {
            "feature_key": feature_key,
            "group_by": group_by,
            "bins": bins,
            "mode": mode,
            "range": None,
            "groups": [],
            "stats": summarize_feature_values(records, feature_key),
        }

    min_v = min(all_values)
    max_v = max(all_values)
    if max_v <= min_v:
        max_v = min_v + 1e-9
    width = (max_v - min_v) / bins
    groups_payload: list[dict[str, Any]] = []

    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        counts = [0] * bins
        for value in values:
            idx = min(bins - 1, max(0, int((value - min_v) / width)))
            counts[idx] += 1
        payload_bins = [count / sum(counts) if sum(counts) else 0.0 for count in counts] if mode == "density" else counts
        groups_payload.append(
            {
                "group": key,
                "count": len(values),
                "bins": payload_bins,
                "stats": {
                    "min": min(values),
                    "max": max(values),
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                },
            }
        )

    edges = [min_v + width * i for i in range(bins + 1)]
    return {
        "feature_key": feature_key,
        "group_by": group_by,
        "bins": bins,
        "mode": mode,
        "range": {"min": min_v, "max": max_v, "edges": edges},
        "groups": groups_payload,
        "stats": summarize_feature_values(records, feature_key),
    }


def filter_records_by_feature_range(
    records: list[dict[str, Any]],
    *,
    feature_key: str,
    min_value: float | None,
    max_value: float | None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        value = _to_float((record.get("features") or {}).get(feature_key))
        if value is None:
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        out.append(
            {
                "take_id": record.get("take_id"),
                "object_id": record.get("object_id"),
                "dataset_id": record.get("dataset_id"),
                "session_id": record.get("session_id"),
                "physical_object_id": record.get("physical_object_id"),
                "pipeline_id": record.get("pipeline_id"),
                "run_id": record.get("run_id"),
                "stage_id": record.get("stage_id"),
                "labeled_superclass": record.get("labeled_superclass"),
                "labeled_superclass_source": record.get("labeled_superclass_source"),
                "labeled_superclass_unresolved": record.get("labeled_superclass_unresolved"),
                "normalized_class": record.get("normalized_class"),
                "processed_superclass": record.get("processed_superclass"),
                "processed_class": record.get("processed_class"),
                "processed_label": record.get("processed_label"),
                "confidence": record.get("confidence"),
                "superclass": record.get("processed_superclass"),
                "raw_labels": record.get("raw_labels"),
                "labels": record.get("labels"),
                "validation_status": record.get("validation_status"),
                "split": record.get("split"),
                "feature_value": value,
                "feature_source": (record.get("feature_sources") or {}).get(feature_key),
                "timestamp": record.get("timestamp"),
                "annotation": record.get("annotation"),
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


def summarize_records_by_feature_range(
    records: list[dict[str, Any]],
    *,
    feature_key: str,
    min_value: float | None,
    max_value: float | None,
) -> dict[str, Any]:
    take_ids: set[str] = set()
    physical_object_ids: set[str] = set()
    object_ids: set[str] = set()
    matched_count = 0
    for record in records:
        value = _to_float((record.get("features") or {}).get(feature_key))
        if value is None:
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        matched_count += 1
        take_id = str(record.get("take_id") or "").strip()
        if take_id:
            take_ids.add(take_id)
        physical_object_id = str(record.get("physical_object_id") or "").strip()
        if physical_object_id:
            physical_object_ids.add(physical_object_id)
        object_ids.add(f"{record.get('take_id')}::{record.get('object_id')}")
    return {
        "record_count": matched_count,
        "take_count": len(take_ids),
        "physical_object_count": len(physical_object_ids),
        "object_count": len(object_ids),
    }


def summarize_feature_scope(records: list[dict[str, Any]]) -> dict[str, Any]:
    take_ids = {str(item.get("take_id") or "") for item in records if str(item.get("take_id") or "")}
    physical_object_ids = {
        str(item.get("physical_object_id") or "")
        for item in records
        if str(item.get("physical_object_id") or "")
    }
    return {
        "record_count": len(records),
        "take_count": len(take_ids),
        "physical_object_count": len(physical_object_ids),
        "object_count": len({f"{item.get('take_id')}::{item.get('object_id')}" for item in records}),
    }


def build_filter_options(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_labels: set[str] = set()
    normalized_classes: set[str] = set()
    labeled_superclasses: set[str] = set()
    processed_superclasses: set[str] = set()
    physical_object_ids: set[str] = set()
    take_ids: set[str] = set()
    validation_statuses: set[str] = set()
    splits: set[str] = set()
    pipeline_ids: set[str] = set()
    calibration_ids: set[str] = set()
    processed_classes: set[str] = set()

    for record in records:
        take_id = str(record.get("take_id") or "").strip()
        if take_id:
            take_ids.add(take_id)
        physical_object_id = str(record.get("physical_object_id") or "").strip()
        if physical_object_id:
            physical_object_ids.add(physical_object_id)
        normalized_class = str(record.get("normalized_class") or "").strip()
        if normalized_class:
            normalized_classes.add(normalized_class)
        labeled_superclass = str(record.get("labeled_superclass") or "").strip()
        if labeled_superclass:
            labeled_superclasses.add(labeled_superclass)
        processed_superclass = str(record.get("processed_superclass") or "").strip()
        if processed_superclass:
            processed_superclasses.add(processed_superclass)
        processed_class = str(record.get("processed_class") or "").strip()
        if processed_class:
            processed_classes.add(processed_class)
        validation_status = str(record.get("validation_status") or "").strip()
        if validation_status:
            validation_statuses.add(validation_status)
        split = str(record.get("split") or "").strip()
        if split:
            splits.add(split)
        pipeline_id = str(record.get("pipeline_id") or "").strip()
        if pipeline_id:
            pipeline_ids.add(pipeline_id)
        calibration_id = str((((record.get("calibration_metadata") or {}) if isinstance(record.get("calibration_metadata"), dict) else {}).get("calibration_id")) or "").strip()
        if calibration_id:
            calibration_ids.add(calibration_id)
        for label in record.get("raw_labels") or []:
            text = str(label).strip()
            if text:
                raw_labels.add(text)

    return {
        "raw_labels": sorted(raw_labels),
        "normalized_classes": sorted(normalized_classes),
        "labeled_superclasses": sorted(labeled_superclasses),
        "processed_superclasses": sorted(processed_superclasses),
        "superclasses": sorted(processed_superclasses),
        "processed_classes": sorted(processed_classes),
        "physical_object_ids": sorted(physical_object_ids),
        "take_ids": sorted(take_ids),
        "validation_statuses": sorted(validation_statuses),
        "splits": sorted(splits),
        "pipeline_ids": sorted(pipeline_ids),
        "calibration_ids": sorted(calibration_ids),
        "stats": summarize_feature_scope(records),
    }
