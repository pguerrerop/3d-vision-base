from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.filesystem import get_take_detail, list_takes
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.service import _extract_scalar_features


def _make_settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = _mean(xs), _mean(ys)
    if mx is None or my is None:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den <= 0:
        return None
    return num / den


def _classify_cv(cv: float | None) -> str:
    if cv is None:
        return "unknown"
    if cv < 0.03:
        return "low_sensitivity"
    if cv < 0.08:
        return "medium_sensitivity"
    return "high_sensitivity"


def _collect_take_ids(service: DatasetService, settings: ApiSettings, *, dataset_id: str | None, ml_set_id: str | None) -> tuple[str | None, list[str]]:
    if ml_set_id:
        resolved = service.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
        did = str(resolved.get("dataset_id") or "")
        memberships = service.list_ml_set_memberships(dataset_id=did, ml_set_id=ml_set_id)
        take_ids = sorted({str(item.get("take_id") or "") for item in memberships if bool(item.get("include", True)) and str(item.get("take_id") or "")})
        return did, take_ids
    if not dataset_id:
        raise ValueError("dataset_id is required when ml_set_id is not provided")
    takes = list_takes(settings, dataset_id=dataset_id, show_archived=False)
    return dataset_id, sorted({t.take_id for t in takes})


def analyze_repeatability(
    *,
    data_dir: Path,
    output_dir: Path,
    dataset_id: str | None = None,
    ml_set_id: str | None = None,
    pipeline_id: str | None = None,
    include_diagnostics: bool = False,
    include_invalidity_flags: bool = False,
    include_provenance_summaries: bool = False,
) -> dict[str, Any]:
    service = DatasetService(data_dir)
    settings = _make_settings(data_dir)
    resolved_dataset_id, take_ids = _collect_take_ids(service, settings, dataset_id=dataset_id, ml_set_id=ml_set_id)
    if not take_ids:
        raise ValueError("no takes selected for analysis")

    object_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_processing: list[str] = []
    rule_set_counter: Counter[tuple[str, str, str]] = Counter()
    for take_id in take_ids:
        detail = get_take_detail(settings, take_id)
        if detail is None:
            continue
        meta = detail.take_metadata if isinstance(detail.take_metadata, dict) else {}
        obj_id = str(meta.get("physical_object_id") or "").strip()
        if not obj_id:
            obj_id = f"__take__:{take_id}"
        result = detail.result if isinstance(detail.result, dict) else None
        if not result:
            missing_processing.append(take_id)
            continue
        if pipeline_id:
            result_pipeline_id = str(((result.get("processing_pipeline") or {}).get("id") or "")).strip()
            if result_pipeline_id != pipeline_id:
                missing_processing.append(take_id)
                continue
        cls = result.get("classification") if isinstance(result.get("classification"), dict) else {}
        rule_set_counter[
            (
                str(cls.get("rule_set_id") or "unknown"),
                str(cls.get("rule_set_version") or "unknown"),
                str(cls.get("rule_set_source") or "unknown"),
            )
        ] += 1
        features = _extract_scalar_features(result)
        if include_diagnostics:
            diag_path = data_dir / "processed" / take_id / "feature_vector.json"
            if diag_path.is_file():
                payload = json.loads(diag_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("features"), dict):
                    for key, value in payload["features"].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            features[f"diag_{key}"] = float(value)
        object_rows[obj_id].append({"take_id": take_id, "features": features})

    feature_by_object: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for object_id, rows in object_rows.items():
        for row in rows:
            for key, value in row["features"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_by_object[object_id][key].append(float(value))

    per_object_feature_stats: list[dict[str, Any]] = []
    for object_id in sorted(feature_by_object.keys()):
        for feature in sorted(feature_by_object[object_id].keys()):
            values = feature_by_object[object_id][feature]
            mu = _mean(values)
            sd = _std(values)
            cv = (sd / abs(mu)) if (mu is not None and sd is not None and abs(mu) > 1e-12) else None
            outlier_count = 0
            if len(values) >= 3 and mu is not None and sd is not None and sd > 0:
                outlier_count = sum(1 for v in values if abs(v - mu) > (2.5 * sd))
            per_object_feature_stats.append(
                {
                    "physical_object_id": object_id,
                    "feature": feature,
                    "take_count": len(values),
                    "mean": mu,
                    "std": sd,
                    "cv": cv,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "outlier_count": outlier_count,
                    "missing_data_ratio": 0.0,
                    "orientation_sensitivity": _classify_cv(cv),
                }
            )

    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_object_feature_stats:
        by_feature[str(row["feature"])].append(row)

    feature_stability: list[dict[str, Any]] = []
    for feature in sorted(by_feature.keys()):
        cvs = [float(row["cv"]) for row in by_feature[feature] if isinstance(row.get("cv"), (int, float))]
        mean_cv = _mean(cvs)
        feature_stability.append(
            {
                "feature": feature,
                "average_cv": mean_cv,
                "object_count": len(by_feature[feature]),
                "instability_flag": bool(mean_cv is not None and mean_cv >= 0.08),
                "orientation_sensitivity": _classify_cv(mean_cv),
            }
        )
    feature_stability.sort(key=lambda row: (float(row["average_cv"]) if isinstance(row["average_cv"], (int, float)) else 1e9), reverse=True)

    quality_feature_candidates = (
        "summary_valid_pixel_ratio",
        "summary_invalid_pixel_ratio",
        "summary_border_touch_ratio",
        "summary_segmentation_foreground_ratio",
    )
    correlations: list[dict[str, Any]] = []
    for quality_feature in quality_feature_candidates:
        quality_values: list[float] = []
        cv_values: list[float] = []
        for row in per_object_feature_stats:
            feature_name = str(row["feature"])
            if feature_name == quality_feature:
                continue
            cv = row.get("cv")
            if not isinstance(cv, (int, float)):
                continue
            object_id = str(row["physical_object_id"])
            q_rows = [item for item in per_object_feature_stats if str(item["physical_object_id"]) == object_id and str(item["feature"]) == quality_feature]
            if not q_rows:
                continue
            q_mean = q_rows[0].get("mean")
            if not isinstance(q_mean, (int, float)):
                continue
            quality_values.append(float(q_mean))
            cv_values.append(float(cv))
        corr = _pearson(quality_values, cv_values)
        correlations.append({"quality_feature": quality_feature, "cv_correlation": corr, "sample_count": len(quality_values)})

    quality_associations: list[dict[str, Any]] = []
    if include_invalidity_flags:
        flag_counts_by_object: dict[str, list[int]] = defaultdict(list)
        for object_id, rows in object_rows.items():
            for row in rows:
                flags_path = data_dir / "processed" / str(row["take_id"]) / "quality_flags.json"
                count = 0
                if flags_path.is_file():
                    payload = json.loads(flags_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and isinstance(payload.get("flags"), list):
                        count = len(payload["flags"])
                flag_counts_by_object[object_id].append(count)
        for object_id, counts in flag_counts_by_object.items():
            quality_associations.append(
                {
                    "physical_object_id": object_id,
                    "mean_flag_count": _mean([float(v) for v in counts]),
                    "max_flag_count": max(counts) if counts else 0,
                }
            )

    provenance_validity: list[dict[str, Any]] = []
    if include_provenance_summaries:
        validity_counter: Counter[tuple[str, str]] = Counter()
        for take_id in take_ids:
            prov_path = data_dir / "processed" / take_id / "feature_provenance.json"
            if not prov_path.is_file():
                continue
            payload = json.loads(prov_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            for feat, meta in payload.items():
                if not isinstance(meta, dict):
                    continue
                validity = str(meta.get("validity") or "unknown")
                validity_counter[(str(feat), validity)] += 1
        for (feature, validity), count in sorted(validity_counter.items()):
            provenance_validity.append({"feature": feature, "validity": validity, "count": int(count)})

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": resolved_dataset_id,
        "ml_set_id": ml_set_id,
        "pipeline_id": pipeline_id,
        "take_count_requested": len(take_ids),
        "take_count_analyzed": sum(len(rows) for rows in object_rows.values()),
        "physical_object_count": len(object_rows),
        "missing_processing_take_ids": sorted(set(missing_processing)),
        "rule_set_distribution": [
            {
                "rule_set_id": rid,
                "rule_set_version": ver,
                "rule_set_source": src,
                "take_count": int(count),
            }
            for (rid, ver, src), count in sorted(rule_set_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]))
        ],
        "feature_stability": feature_stability,
        "correlations": correlations,
        "quality_associations": quality_associations,
        "provenance_validity": provenance_validity,
    }

    (output_dir / "repeatability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output_dir / "repeatability_per_object_feature.csv", per_object_feature_stats)
    _write_csv(output_dir / "repeatability_feature_stability.csv", feature_stability)
    _write_csv(output_dir / "repeatability_correlations.csv", correlations)
    if quality_associations:
        _write_csv(output_dir / "repeatability_quality_associations.csv", quality_associations)
    if provenance_validity:
        _write_csv(output_dir / "repeatability_provenance_validity.csv", provenance_validity)

    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze feature repeatability across repeated takes grouped by physical_object_id.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for summary/CSV artifacts")
    parser.add_argument("--dataset-id", type=str, default=None, help="Dataset scope when analyzing by dataset or disambiguating ml_set_id")
    parser.add_argument("--ml-set-id", type=str, default=None, help="ML set scope (included memberships only)")
    parser.add_argument("--pipeline-id", type=str, default=None, help="Optional pipeline id filter (example: mining_steel_ball_classification_25d)")
    parser.add_argument("--include-diagnostics", action="store_true", help="Include persisted diagnostics feature_vector fields in analysis")
    parser.add_argument("--include-invalidity-flags", action="store_true", help="Include quality flag associations in summary")
    parser.add_argument("--include-provenance-summaries", action="store_true", help="Include feature provenance validity statistics")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (args.data_dir / "runtime" / "analysis" / f"repeatability_{ts}")
    summary = analyze_repeatability(
        data_dir=args.data_dir,
        output_dir=output_dir,
        dataset_id=args.dataset_id,
        ml_set_id=args.ml_set_id,
        pipeline_id=args.pipeline_id,
        include_diagnostics=bool(args.include_diagnostics),
        include_invalidity_flags=bool(args.include_invalidity_flags),
        include_provenance_summaries=bool(args.include_provenance_summaries),
    )
    print("Repeatability analysis complete")
    print(f"- output_dir: {output_dir}")
    print(f"- take_count_analyzed: {summary['take_count_analyzed']}")
    print(f"- physical_object_count: {summary['physical_object_count']}")
    print(f"- missing_processing: {len(summary['missing_processing_take_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
