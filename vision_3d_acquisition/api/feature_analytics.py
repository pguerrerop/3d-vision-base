from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
import time
from typing import Any

from vision_3d_acquisition.api.filesystem import list_take_ids, read_json
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


@dataclass(frozen=True)
class FeatureDefinition:
    feature_key: str
    display_name: str
    semantic_group: str
    unit: str | None
    description: str
    source_stage: str
    source_metric_path: str


KNOWN_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition("feature_eccentricity", "Eccentricity", "geometry", None, "Ellipse eccentricity", "measurement", "objects[].feature_eccentricity"),
    FeatureDefinition("feature_sphericity_3d", "Sphericity 3D", "morphology", None, "3D sphericity score", "measurement", "objects[].feature_sphericity_3d"),
    FeatureDefinition("feature_flatness", "Flatness", "morphology", None, "Object flatness indicator", "measurement", "objects[].feature_flatness"),
    FeatureDefinition("feature_edge_roughness", "Edge roughness", "morphology", None, "Boundary roughness metric", "measurement", "objects[].feature_edge_roughness"),
    FeatureDefinition("feature_volume_proxy_mm3", "Volume proxy", "geometry", "mm3", "Proxy volume from object geometry", "measurement", "objects[].feature_volume_proxy_mm3"),
    FeatureDefinition("height_max_mm", "Max height", "height", "mm", "Max height above reference", "measurement", "objects[].height_above_belt_mm.max"),
    FeatureDefinition("height_mean_mm", "Mean height", "height", "mm", "Mean height above reference", "measurement", "objects[].height_above_belt_mm.mean"),
    FeatureDefinition("fit_rmse_mm", "Surface fit RMSE", "morphology", "mm", "Surface fit residual", "classification", "objects[].fit_rmse_mm"),
    FeatureDefinition("footprint_roundness", "Footprint roundness", "geometry", None, "Roundness score on footprint", "measurement", "objects[].feature_footprint_roundness"),
    FeatureDefinition("confidence", "Classification confidence", "classification", None, "Classifier confidence score", "classification", "objects[].confidence"),
]


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


def _extract_feature_values(raw_object: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw_object.items():
        if key.startswith("feature_"):
            parsed = _to_float(value)
            if parsed is not None:
                out[key] = parsed
    height = raw_object.get("height_above_belt_mm")
    if isinstance(height, dict):
        max_h = _to_float(height.get("max") if "max" in height else height.get("max_height_mm"))
        mean_h = _to_float(height.get("mean") if "mean" in height else height.get("mean_height_mm"))
        if max_h is not None:
            out["height_max_mm"] = max_h
        if mean_h is not None:
            out["height_mean_mm"] = mean_h
    fit_rmse = _to_float(raw_object.get("fit_rmse_mm"))
    if fit_rmse is not None:
        out["fit_rmse_mm"] = fit_rmse
    roundness = _to_float(raw_object.get("feature_footprint_roundness") or raw_object.get("footprint_roundness") or raw_object.get("sphericity_score"))
    if roundness is not None:
        out["footprint_roundness"] = roundness
    confidence = _to_float(raw_object.get("confidence"))
    if confidence is not None:
        out["confidence"] = confidence
    return out


def _feature_definitions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {item.feature_key: item for item in KNOWN_FEATURES}
    found: set[str] = set()
    for record in records:
        for key in (record.get("features") or {}).keys():
            found.add(str(key))
    output: list[dict[str, Any]] = []
    for key in sorted(found):
        if key in known:
            entry = known[key]
            output.append(
                {
                    "feature_key": entry.feature_key,
                    "display_name": entry.display_name,
                    "semantic_group": entry.semantic_group,
                    "unit": entry.unit,
                    "description": entry.description,
                    "source_stage": entry.source_stage,
                    "source_metric_path": entry.source_metric_path,
                }
            )
        else:
            output.append(
                {
                    "feature_key": key,
                    "display_name": key.replace("_", " "),
                    "semantic_group": "runtime" if key.startswith("runtime_") else "measurement",
                    "unit": None,
                    "description": "Derived from object metrics",
                    "source_stage": "measurement",
                    "source_metric_path": f"objects[].{key}",
                }
            )
    return output


def build_feature_records(
    settings: ApiSettings,
    *,
    query: dict[str, Any] | None = None,
    max_takes: int = 600,
    time_budget_ms: int = 2500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    filters = query or {}
    dataset_ids = set(_split_csv(filters.get("datasets")))
    session_ids = set(_split_csv(filters.get("sessions")))
    labels_filter = {item.lower() for item in _split_csv(filters.get("labels"))}
    superclass_filter = {item.upper() for item in _split_csv(filters.get("superclass"))}
    validation_filter = {item.lower() for item in _split_csv(filters.get("validation_status"))}
    split_filter = {item.lower() for item in _split_csv(filters.get("split"))}
    pipeline_filter = set(_split_csv(filters.get("pipeline")))
    calibration_filter = set(_split_csv(filters.get("calibration")))
    selected_features = set(_split_csv(filters.get("feature_selection")))
    from_date = _parse_date(filters.get("date_from"))
    to_date = _parse_date(filters.get("date_to"))

    dataset_service = DatasetService(settings.data_dir)

    take_ids = sorted(list_take_ids(settings), reverse=True)
    records: list[dict[str, Any]] = []
    scan_meta = {
        "take_candidates": len(take_ids),
        "takes_scanned": 0,
        "takes_with_result": 0,
        "objects_scanned": 0,
        "records_emitted": 0,
        "stopped_by": None,
        "max_takes": max(1, int(max_takes)),
        "time_budget_ms": max(50, int(time_budget_ms)),
    }

    for take_id in take_ids:
        elapsed_ms = (time.monotonic() - started) * 1000.0
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
        if from_date and (created_date is None or created_date < from_date):
            continue
        if to_date and (created_date is None or created_date > to_date):
            continue
        if split_filter and split.lower() not in split_filter:
            continue

        processing_pipeline = result.get("processing_pipeline") if isinstance(result.get("processing_pipeline"), dict) else {}
        pipeline_id = str(processing_pipeline.get("id") or "")
        run_id = str(result.get("run_id") or "")
        calibration_id = str(result.get("calibration_id") or "")

        if pipeline_filter and pipeline_id not in pipeline_filter:
            continue
        if calibration_filter and calibration_id not in calibration_filter:
            continue

        objects = [
            item
            for item in ((result.get("objects") or []) + (result.get("rejected_objects") or []))
            if isinstance(item, dict)
        ]

        superclasses = [str(item).upper() for item in (take_management.get("superclass_labels") or []) if str(item)]
        fallback_superclass = superclasses[0] if superclasses else str(result.get("processed_superclass") or "UNKNOWN")

        for raw in objects:
            scan_meta["objects_scanned"] += 1
            features = _extract_feature_values(raw)
            if selected_features:
                features = {key: value for key, value in features.items() if key in selected_features}
            if not features:
                continue

            labels = [str(raw.get("class_name") or "")] + [str(item) for item in (take_management.get("semantic_labels") or [])]
            labels = [item for item in labels if item]
            if labels_filter and not any(item.lower() in labels_filter for item in labels):
                continue

            superclass = _classify_superclass(raw, fallback_superclass)
            if superclass_filter and superclass not in superclass_filter:
                continue

            records.append(
                {
                    "object_id": str(raw.get("object_id") or ""),
                    "take_id": take_id,
                    "dataset_id": dataset_id,
                    "session_id": session_id,
                    "physical_object_id": str(take_management.get("physical_object_id") or "") or None,
                    "pipeline_id": pipeline_id,
                    "run_id": run_id,
                    "stage_id": "measurement",
                    "labels": labels,
                    "superclass": superclass,
                    "features": features,
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
                    "split": split or None,
                    "annotation": None,
                }
            )

    scan_meta["records_emitted"] = len(records)
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
        if group_by == "label":
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
                "pipeline_id": record.get("pipeline_id"),
                "run_id": record.get("run_id"),
                "stage_id": record.get("stage_id"),
                "superclass": record.get("superclass"),
                "labels": record.get("labels"),
                "feature_value": value,
                "timestamp": record.get("timestamp"),
                "annotation": record.get("annotation"),
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out
